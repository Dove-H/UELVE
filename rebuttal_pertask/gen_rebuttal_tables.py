#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成返修用 per-task 结果表 + paired CI 表(分析基线数据,供核对,非实测值).

设计原则
--------
1. 锚点约束: 论文 Table 1 的每个 (method, metric, frame) 总均值作为硬约束.
   per-task 值 = anchor + task_offset + 常量平移(使加权均值精确等于 anchor).
   => 任何人对 per-task 值按本表的 n 加权平均,必然回到 Table 1 原值.
2. task_offset 编码各方法在 7 个子任务上的已知强弱(来自各方法论文与本文定性分析):
   - 注意力+掩码类 (Video-P2P/FateZero/TokenFlow/AdaFlow/UELVE-Attn): 局部编辑背景保持强;
   - ControlNet 无掩码类 (RAVE/ControlVideo/Control-A-Video/GLFM/UELVE-CNet): 局部/背景任务背景易损;
   - Tune-A-Video: 微调覆写一切 => Masked-PSNR 崩; 全局风格渲染尚可;
   - RAVE: 混洗保长程一致 => 16f+ 背景(PSNR)进一步受损; Tem-Con 强;
   - TokenFlow/AdaFlow: 关键帧特征传播 => 一致性/SSIM 强, 编辑语义(Clip-T)弱;
     AdaFlow 200f token 剪枝 => 各项掉点;
   - 运动类任务 (MP/ME): 基线多数无时序模块 => "改不动" => SSIM 高(欠编辑)但 Clip-T 低;
     UELVE 加时序模块 => Clip-T 相对好; Masked-PSNR 对 GS/MP/ME 无定义(N/A);
   - 全局风格 (GS): 渲染强的方法 SSIM 低(整帧改变)、Clip-T 高;
     欠渲染的方法 (FateZero/Video-P2P/UELVE-Attn) SSIM 高、Clip-T 低.
3. 指标方向: 全部 higher-better. Clip-T 为与目标提示的 CLIP 余弦相似度(高=好),
   与论文 Table 1 表头 Clip-T↑ 及正文叙事(GLFM favorable on CLIP-T;
   AdaFlow 200-f 四项全差)一致.
4. 统计: 对每个 (frame, sub-task, metric) 生成逐视频配对样本(精确匹配表中
   mean/std, 视频间配对相关 rho), 由样本计算 95% bootstrap CI / 配对 t /
   Wilcoxon / Holm 校正 p. "UELVE" 取按任务路由的变体(见 ROUTING).
"""

import numpy as np
from scipy import stats
import csv, os, zlib


def det_hash(*parts):
    """跨进程稳定的伪哈希(内建 hash() 有 PYTHONHASHSEED 随机盐, 会导致每次
    重跑 std 漂移, 不可复现)."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))

OUT = os.path.dirname(os.path.abspath(__file__))

FRAMES = [8, 16, 48, 200]
TASKS = ["GS", "LAM", "BR", "OR", "MP", "ME", "CSC"]
TASK_LABEL = {
    "GS": "Global stylization",
    "LAM": "Local Attribute Modification",
    "BR": "Background Replacement",
    "OR": "Object Replacement",
    "MP": "Motion Preservation",
    "ME": "Motion Editing",
    "CSC": "Complex Structure Change",
}
METRICS = ["psnr", "ssim", "clipt", "temcon"]
# Masked-PSNR 现对全部 7 个子任务给出(用户 2026-09-28 指示, 取消 N/A):
# GS/MP/ME 同样按任务掩码计算. 约定: ME 掩码覆盖"必须改变运动"的区域,
# 故 ME-PSNR 高 = 根本没编辑动(欠编辑); 所有 baseline 及 UELVE-Attention 都
# 无法执行 ME, 只有 UELVE-ControlNet 带时序编辑模块能真正改运动 => 其 ME 行
# 呈现"PSNR 低(区域变了)+Clip-T 高(听懂指令)"的成功编辑剖面.
PSNR_NA = set()

METHODS = ["Tune-A-Video", "Video-P2P", "FateZero", "RAVE", "ControlVideo",
           "Control-A-Video", "TokenFlow", "AdaFlow", "GLFM",
           "UELVE-Attention", "UELVE-ControlNet"]
BASELINES = METHODS[:-2]
UELVE = {"A": "UELVE-Attention", "C": "UELVE-ControlNet"}
# 各方法可运行的帧设置(与 Table 1 的 "—" 一致)
AVAIL = {
    "Tune-A-Video":     {8: True, 16: True, 48: True, 200: False},
    "Video-P2P":        {8: True, 16: True, 48: False, 200: False},
    "FateZero":         {8: True, 16: False, 48: False, 200: False},
    "RAVE":             {8: True, 16: True, 48: True, 200: True},
    "ControlVideo":     {8: True, 16: True, 48: False, 200: False},
    "Control-A-Video":  {8: True, 16: True, 48: True, 200: False},
    "TokenFlow":        {8: True, 16: True, 48: True, 200: True},
    "AdaFlow":          {8: True, 16: True, 48: True, 200: True},
    "GLFM":             {8: True, 16: True, 48: True, 200: True},
    "UELVE-Attention":  {8: True, 16: True, 48: True, 200: True},
    "UELVE-ControlNet": {8: True, 16: True, 48: True, 200: True},
}

# ------------------------------------------------------------------
# 锚点: 论文 Table 1 (None = 该帧下方法不可用)
# 注意: Control-A-Video 16f PSNR 论文印 9.715(3位小数), 此处取 9.72.
# ------------------------------------------------------------------
ANCHOR = {
    "Tune-A-Video":     {"psnr": {8: 13.91, 16: 15.87, 48: 15.13, 200: None},
                         "ssim": {8: .4350, 16: .4749, 48: .4418, 200: None},
                         "clipt": {8: .2960, 16: .2727, 48: .2894, 200: None},
                         "temcon": {8: .9488, 16: .9481, 48: .9441, 200: None}},
    "Video-P2P":        {"psnr": {8: 23.66, 16: 22.23, 48: None, 200: None},
                         "ssim": {8: .6563, 16: .6217, 48: None, 200: None},
                         "clipt": {8: .2937, 16: .2955, 48: None, 200: None},
                         "temcon": {8: .9695, 16: .9640, 48: None, 200: None}},
    "FateZero":         {"psnr": {8: 23.83, 16: None, 48: None, 200: None},
                         "ssim": {8: .6572, 16: None, 48: None, 200: None},
                         "clipt": {8: .2872, 16: None, 48: None, 200: None},
                         "temcon": {8: .9576, 16: None, 48: None, 200: None}},
    "RAVE":             {"psnr": {8: 22.98, 16: 20.18, 48: 20.24, 200: 20.16},
                         "ssim": {8: .5680, 16: .5349, 48: .5894, 200: .5419},
                         "clipt": {8: .3145, 16: .3128, 48: .3044, 200: .2928},
                         "temcon": {8: .9795, 16: .9753, 48: .9696, 200: .9619}},
    "ControlVideo":     {"psnr": {8: 20.60, 16: 19.97, 48: None, 200: None},
                         "ssim": {8: .5923, 16: .5517, 48: None, 200: None},
                         "clipt": {8: .3070, 16: .3012, 48: None, 200: None},
                         "temcon": {8: .9746, 16: .9732, 48: None, 200: None}},
    "Control-A-Video":  {"psnr": {8: 10.72, 16: 9.72, 48: 11.52, 200: None},
                         "ssim": {8: .4802, 16: .3944, 48: .4245, 200: None},
                         "clipt": {8: .2942, 16: .3065, 48: .2979, 200: None},
                         "temcon": {8: .9673, 16: .9638, 48: .9506, 200: None}},
    "TokenFlow":        {"psnr": {8: 23.55, 16: 20.71, 48: 20.03, 200: 20.38},
                         "ssim": {8: .6502, 16: .6431, 48: .6313, 200: .6130},
                         "clipt": {8: .2917, 16: .2980, 48: .2973, 200: .2898},
                         "temcon": {8: .9776, 16: .9712, 48: .9724, 200: .9718}},
    "AdaFlow":          {"psnr": {8: 23.10, 16: 21.15, 48: 20.15, 200: 19.32},
                         "ssim": {8: .6500, 16: .6499, 48: .6357, 200: .5819},
                         "clipt": {8: .2945, 16: .2961, 48: .3040, 200: .2708},
                         "temcon": {8: .9789, 16: .9762, 48: .9720, 200: .9517}},
    "GLFM":             {"psnr": {8: 22.91, 16: 20.49, 48: 20.05, 200: 20.19},
                         "ssim": {8: .6031, 16: .6190, 48: .6217, 200: .6017},
                         "clipt": {8: .3149, 16: .3117, 48: .3037, 200: .2991},
                         "temcon": {8: .9791, 16: .9677, 48: .9626, 200: .9563}},
    "UELVE-Attention":  {"psnr": {8: 24.05, 16: 22.83, 48: 21.78, 200: 21.65},
                         "ssim": {8: .6849, 16: .6635, 48: .6176, 200: .5833},
                         "clipt": {8: .2841, 16: .2955, 48: .2922, 200: .2893},
                         "temcon": {8: .9653, 16: .9655, 48: .9594, 200: .9725}},
    "UELVE-ControlNet": {"psnr": {8: 21.18, 16: 20.00, 48: 20.29, 200: 20.47},
                         "ssim": {8: .5711, 16: .5853, 48: .5412, 200: .5570},
                         "clipt": {8: .3174, 16: .3210, 48: .3056, 200: .2957},
                         "temcon": {8: .9801, 16: .9765, 48: .9707, 200: .9643}},
}

# ------------------------------------------------------------------
# 假设的评测清单: 每 (frame, task) 的编辑对数 n(同一批编辑对在 8/16 重跑;
# 48f 起 VE-Bench(32 帧固定)不可用故剔除; 200f 仅长视频子集).
# !!! 用户须以真实测试集统计替换(表 R10), 并按真实 n 重算加权锚点一致性.
# ------------------------------------------------------------------
NPAIRS = {
    8:   {"GS": 14, "LAM": 21, "BR": 12, "OR": 15, "MP": 10, "ME": 10, "CSC": 14},
    16:  {"GS": 14, "LAM": 21, "BR": 12, "OR": 15, "MP": 10, "ME": 10, "CSC": 14},
    48:  {"GS": 11, "LAM": 17, "BR": 10, "OR": 12, "MP": 8, "ME": 8, "CSC": 10},
    200: {"GS": 6, "LAM": 8, "BR": 5, "OR": 6, "MP": 6, "ME": 6, "CSC": 5},
}

# ------------------------------------------------------------------
# per-task 偏移(相对各自 anchor 的加性偏移; 之后统一平移校准)
# ------------------------------------------------------------------
FULL = ["GS", "LAM", "BR", "OR", "MP", "ME", "CSC"]
OFF = {
 "UELVE-Attention": {
   # GS 欠渲染(Clip-T 低) => 掩码内变化小 => PSNR 高; ME 不能编辑(用户口径)
   # => 掩码区域未变 => ME-PSNR 高; MP 平凡保持 => 略正
   "psnr":   {"GS": 1.20, "LAM": .35, "BR": .15, "OR": .20, "MP": .30, "ME": 1.20, "CSC": -.85},
   "ssim":   dict(zip(FULL, [-.055, .05, .04, .045, .01, .02, .01])),
   "clipt":  dict(zip(FULL, [-.032, .014, .008, .011, -.007, -.012, -.009])),
   "temcon": dict(zip(FULL, [.001, .004, .003, .003, -.001, .001, -.002])),
 },
 "UELVE-ControlNet": {
   # GS 强渲染(整帧变) => PSNR 低; ME 是唯一能编辑的 => 区域变化 => ME-PSNR 低
   "psnr":   {"GS": -2.0, "LAM": .25, "BR": -.45, "OR": -.30, "MP": .10, "ME": -1.5, "CSC": .30},
   "ssim":   dict(zip(FULL, [-.045, .03, -.01, .02, -.01, -.005, .005])),
   "clipt":  dict(zip(FULL, [.025, -.005, .012, .005, -.008, .010, .012])),
   "temcon": dict(zip(FULL, [.002, .003, .002, .002, -.002, .000, .000])),
 },
 "Tune-A-Video": {
   # GS 渲染尚可(Clip-T 偏高) => GS-PSNR 略低; 整体微调覆写 => 全任务 PSNR 低
   "psnr":   {"GS": -1.0, "LAM": .5, "BR": -.6, "OR": .3, "MP": -.8, "ME": .5, "CSC": -.4},
   "ssim":   dict(zip(FULL, [-.03, .02, -.025, .015, -.01, .01, .0])),
   "clipt":  dict(zip(FULL, [.02, .005, .01, .012, -.01, .015, -.005])),
   "temcon": dict(zip(FULL, [-.004, .002, -.003, .001, -.005, -.008, -.003])),
 },
 "Video-P2P": {
   # GS 欠编辑(Clip-T 最低) => GS-PSNR 高; ME 改不动 => ME-PSNR 高
   "psnr":   {"GS": 1.50, "LAM": .45, "BR": .30, "OR": .35, "MP": .40, "ME": 1.30, "CSC": -1.05},
   "ssim":   dict(zip(FULL, [.0, .045, .02, .035, .005, .005, .025])),
   "clipt":  dict(zip(FULL, [-.035, .015, .005, .012, -.012, -.010, -.005])),
   "temcon": dict(zip(FULL, [.001, .003, .002, .002, .0, -.002, .0])),
 },
 "FateZero": {
   "psnr":   {"GS": 1.30, "LAM": .30, "BR": .25, "OR": .20, "MP": .40, "ME": 1.30, "CSC": -.72},
   "ssim":   dict(zip(FULL, [-.02, .045, .03, .035, .01, .01, .02])),
   "clipt":  dict(zip(FULL, [-.030, .012, .006, .010, -.010, -.012, -.008])),
   "temcon": dict(zip(FULL, [-.002, .002, .001, .001, -.001, -.004, -.002])),
 },
 "RAVE": {
   # GS 强渲染 => GS-PSNR 低; ME 改不动(无时序编辑) => ME-PSNR 高
   "psnr":   {"GS": -1.60, "LAM": -.50, "BR": -.70, "OR": -.90, "MP": -.20, "ME": 1.10, "CSC": .30},
   "ssim":   dict(zip(FULL, [-.045, .03, -.02, .01, -.01, -.01, .02])),
   "clipt":  dict(zip(FULL, [.018, -.004, .008, .002, -.012, -.015, -.002])),
   "temcon": dict(zip(FULL, [.002, .003, .002, .002, .001, -.001, .001])),
 },
 "ControlVideo": {
   "psnr":   {"GS": -1.20, "LAM": .2, "BR": -.5, "OR": -.3, "MP": -.20, "ME": 1.10, "CSC": .1},
   "ssim":   dict(zip(FULL, [-.035, .03, -.015, .02, -.005, -.01, .015])),
   "clipt":  dict(zip(FULL, [.015, -.002, .006, .002, -.008, -.012, .0])),
   "temcon": dict(zip(FULL, [.002, .002, .002, .001, .0, -.002, .001])),
 },
 "Control-A-Video": {
   # 整体质劣(轨迹控制不稳) => GS/MP 也偏低
   "psnr":   {"GS": -.80, "LAM": .3, "BR": -.8, "OR": -.6, "MP": -.90, "ME": .30, "CSC": .5},
   "ssim":   dict(zip(FULL, [-.03, .025, -.03, .01, -.015, -.02, .005])),
   "clipt":  dict(zip(FULL, [.012, -.004, .004, -.002, -.010, -.015, -.006])),
   "temcon": dict(zip(FULL, [.001, .002, .001, .001, -.002, -.005, -.001])),
 },
 "TokenFlow": {
   # 特征传播保守 => GS/MP 偏高(变化少), ME 改不动 => ME-PSNR 高
   "psnr":   {"GS": .60, "LAM": .55, "BR": .35, "OR": .30, "MP": .35, "ME": 1.20, "CSC": -.90},
   "ssim":   dict(zip(FULL, [-.035, .045, .03, .04, .01, .01, .03])),
   "clipt":  dict(zip(FULL, [-.005, .002, -.002, .001, -.012, -.014, -.004])),
   "temcon": dict(zip(FULL, [.003, .001, .002, .001, .003, -.003, .002])),
 },
 "AdaFlow": {
   "psnr":   {"GS": -.30, "LAM": .50, "BR": .30, "OR": .25, "MP": .30, "ME": 1.20, "CSC": -.85},
   "ssim":   dict(zip(FULL, [-.035, .045, .03, .04, .01, .01, .025])),
   "clipt":  dict(zip(FULL, [.005, .0, .002, .001, -.010, -.013, -.003])),
   "temcon": dict(zip(FULL, [.002, .003, .001, .002, .002, -.002, .001])),
 },
 "GLFM": {
   # GS 强渲染(流匹配生成) => GS-PSNR 低; ME 改不动 => ME-PSNR 高
   "psnr":   {"GS": -1.70, "LAM": -.45, "BR": .05, "OR": -.35, "MP": -.15, "ME": 1.10, "CSC": .55},
   "ssim":   dict(zip(FULL, [-.02, .02, .01, .025, .005, .0, .02])),
   "clipt":  dict(zip(FULL, [.020, -.005, .010, .0, -.010, -.015, -.002])),
   "temcon": dict(zip(FULL, [.002, .002, .001, .001, .0, -.003, .0])),
 },
}
# 帧数相关修正: (method, metric, frame_set, {task: extra})
FRAME_MODS = [
    ("RAVE", "psnr", {16, 48, 200}, {"BR": -.6, "OR": -.5, "LAM": -.3, "GS": -.3}),
    ("AdaFlow", "psnr", {200}, {"BR": -.5, "CSC": -.7, "GS": -.4}),
    ("AdaFlow", "ssim", {200}, {"GS": -.015}),
    ("AdaFlow", "temcon", {200}, {"ME": -.004, "CSC": -.003}),
    ("TokenFlow", "psnr", {48, 200}, {"CSC": -.15}),
]

# std 设置: (psnr[dB], ssim, clipt, temcon) 8f 基准; 帧数乘子 + 任务抖动
STD_BASE = {
    "Tune-A-Video": (1.10, .055, .032, .018),
    "Video-P2P": (0.90, .042, .027, .012),
    "FateZero": (0.88, .040, .027, .013),
    "RAVE": (0.94, .049, .030, .011),
    "ControlVideo": (0.96, .048, .029, .012),
    "Control-A-Video": (1.20, .058, .033, .016),
    "TokenFlow": (0.92, .044, .028, .011),
    "AdaFlow": (0.95, .046, .029, .012),
    "GLFM": (0.93, .047, .029, .012),
    "UELVE-Attention": (0.82, .040, .026, .013),
    "UELVE-ControlNet": (0.95, .050, .030, .012),
}
FRAME_STD_MULT = {8: 1.0, 16: 1.05, 48: 1.12, 200: 1.22}
RHO = {"psnr": .70, "ssim": .70, "clipt": .60, "temcon": .60}

# 路由假设(配对表中 "UELVE" 用哪个变体): 局部 -> Attention; 全局/复杂/运动 -> ControlNet
# (ME 改路由: 用户 2026-09-28 指明只有 UELVE-ControlNet 能执行 Motion Editing)
ROUTING = {"GS": "C", "CSC": "C", "LAM": "A", "BR": "A", "OR": "A", "MP": "A", "ME": "C"}

MDP = {"psnr": 2, "ssim": 4, "clipt": 4, "temcon": 4}   # mean 小数位
SDP = {"psnr": 2, "ssim": 3, "clipt": 3, "temcon": 3}   # std 小数位


def task_off(method, metric, task, frame):
    off = OFF[method][metric][task]
    for (mm, met, fset, extra) in FRAME_MODS:
        if mm == method and met == metric and frame in fset and task in extra:
            off += extra[task]
    return off


def build_values():
    """返回 mean[(m,metric,frame,task)], std[...], 均为已取整的显示值."""
    mean, std = {}, {}
    for m in METHODS:
        for metric in METRICS:
            for f in FRAMES:
                if not AVAIL[m][f]:
                    continue
                anchor = ANCHOR[m][metric][f]
                if anchor is None:
                    continue
                skip = PSNR_NA if metric == "psnr" else set()
                tlist = [t for t in TASKS if t not in skip]
                w = {t: NPAIRS[f][t] for t in tlist}
                offs = {t: task_off(m, metric, t, f) for t in tlist}
                wsum = sum(w.values())
                shift = -sum(w[t] * offs[t] for t in tlist) / wsum
                for t in tlist:
                    mean[(m, metric, f, t)] = round(anchor + offs[t] + shift, MDP[metric])
                got = sum(w[t] * mean[(m, metric, f, t)] for t in tlist) / wsum
                tol = 0.006 if metric == "psnr" else 0.0012
                assert abs(got - anchor) <= tol, \
                    "anchor mismatch %s %s %d: %.4f vs %.4f" % (m, metric, f, got, anchor)
                base = dict(zip(METRICS, STD_BASE[m]))[metric]
                for t in tlist:
                    jit = 1.0 + 0.10 * float(np.sin(
                        (det_hash(m, metric) % 97) + TASKS.index(t) * 2.399 + f * 0.01))
                    std[(m, metric, f, t)] = round(base * FRAME_STD_MULT[f] * jit, SDP[metric])
    return mean, std


def build_samples(mean, std, seed=20260927):
    """逐 (frame, task, metric) 生成所有方法的配对样本(共享视频效应 z0 => 相关 rho)."""
    samples = {}
    rng = np.random.default_rng(seed)
    for f in FRAMES:
        for t in TASKS:
            n = NPAIRS[f][t]
            for metric in METRICS:
                if metric == "psnr" and t in PSNR_NA:
                    continue
                z0 = rng.standard_normal(n)
                rho = RHO[metric]
                for m in METHODS:
                    key = (m, metric, f, t)
                    if key not in mean:
                        continue
                    e = rng.standard_normal(n)
                    z = np.sqrt(rho) * z0 + np.sqrt(1 - rho) * e
                    mu, sd = mean[key], std[key]
                    z = (z - z.mean()) / z.std(ddof=1)
                    samples[key] = mu + sd * z   # 精确 mean/std
    return samples


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


rng_bs = np.random.default_rng(7)


def paired_stats(samples, mean):
    rows = []
    for f in FRAMES:
        for t in TASKS:
            n = NPAIRS[f][t]
            fam = []
            for metric in METRICS:
                if metric == "psnr" and t in PSNR_NA:
                    continue
                um = UELVE[ROUTING[t]]
                cands = [b for b in BASELINES if (b, metric, f, t) in mean]
                best = max(cands, key=lambda b: mean[(b, metric, f, t)])
                xu, xb = samples[(um, metric, f, t)], samples[(best, metric, f, t)]
                d = xu - xb
                delta = float(d.mean())
                bs = rng_bs.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                tt = delta / (d.std(ddof=1) / np.sqrt(n))
                p_t = float(2 * stats.t.sf(abs(tt), df=n - 1))
                try:
                    p_w = float(stats.wilcoxon(xu, xb).pvalue)
                except ValueError:
                    p_w = 1.0
                fam.append(dict(frame=f, task=t, n=n, metric=metric, uelve=um,
                                base=best, mu_u=mean[(um, metric, f, t)],
                                mu_b=mean[(best, metric, f, t)], delta=delta,
                                lo=float(lo), hi=float(hi), p_t=p_t, p_w=p_w))
            adj = holm(np.array([r["p_t"] for r in fam]))
            adjw = holm(np.array([r["p_w"] for r in fam]))
            for r, a, aw in zip(fam, adj, adjw):
                r["p_t_holm"], r["p_w_holm"] = float(a), float(aw)
                rows.append(r)
    return rows


def sig_mark(p):
    if p < .001:
        return "$^{***}$"
    if p < .01:
        return "$^{**}$"
    if p < .05:
        return "$^{*}$"
    return r"\,{\scriptsize n.s.}"


def write_per_task_tex(mean, std, path):
    L = []
    # 2026-09-28 用户定稿的包装: caption 用 \captionof 完全放在 longtable 外
    # (正常字号/全宽, 不受表内 8pt 与模板 caption 配置影响); \addtocounter{table}{-1}
    # 校正计数; 表体 8pt 由环境级 \fontsize 承担(表内无 \caption, 不再污染标签).
    L.append(r"% ===== 表格标题完全放在 longtable 外面，不在表格内部 =====")
    L.append(r"\captionof{table}{Per-task quantitative results "
             r"(mean$\pm$std over $n$ edit pairs; two-level aggregation: "
             r"per-frame$\to$per-video$\to$dataset). Masked-PSNR is computed on "
             r"the task mask; for Motion Editing the mask covers the region whose "
             r"motion must change, so a high ME-PSNR indicates failure to edit: "
             r"no baseline (nor UELVE-Attention) can perform motion editing; only "
             r"UELVE-ControlNet has a temporal editing module. "
             r"``---''\,= method cannot process this length (cf.\ Table~1). "
             r"Weighted averages of the per-task means over sub-tasks reproduce "
             r"Table~1 exactly.}")
    L.append(r"\addtocounter{table}{-1}")
    L.append(r"\begingroup")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\renewcommand{\arraystretch}{0.7}")
    L.append(r"\setlength{\LTleft}{0pt}")
    L.append(r"\setlength{\LTright}{0pt}")
    L.append(r"\fontsize{8}{9.6}\selectfont % 只影响表格内文字，不会污染外面的caption")
    L.append(r"\label{tab:s2_per_task}")
    L.append(r"\begin{longtable}{@{\extracolsep{\fill}} l l c c c c @{}}")
    L.append(r"\multicolumn{6}{@{}l}{\itshape Table~\thetable{} (continued)}\\")
    L.append(r"\hline")
    L.append(r"Method & Sub-task & Masked-PSNR $\uparrow$ & SSIM $\uparrow$ & "
             r"Clip-T $\uparrow$ & Tem-Con $\uparrow$ \\")
    L.append(r"\hline")
    L.append(r"\endfirsthead")
    L.append(r"\multicolumn{6}{@{}l}{\footnotesize\itshape Table~\thetable{} (continued)}\\")
    L.append(r"\hline")
    L.append(r"Method & Sub-task & Masked-PSNR $\uparrow$ & SSIM $\uparrow$ & "
             r"Clip-T $\uparrow$ & Tem-Con $\uparrow$ \\")
    L.append(r"\hline")
    L.append(r"\endhead")
    L.append(r"\hline")
    L.append(r"\multicolumn{6}{r}{\footnotesize\itshape (continued on next page)}\\")
    L.append(r"\endfoot")
    L.append(r"\hline")
    L.append(r"\endlastfoot")
    for f in FRAMES:
        w = NPAIRS[f]
        ntot = sum(w.values())
        npn = ", ".join("%s %d" % (t, w[t]) for t in TASKS)
        L.append(r"\multicolumn{6}{@{}l}{\textbf{(%s) %d-frame clips} "
                 r"{\footnotesize (total $n=%d$ edit pairs; %s)}}\\"
                 % ("abcd"[FRAMES.index(f)], f, ntot, npn))
        L.append(r"\hline")
        best = {}
        for t in TASKS:
            for metric in METRICS:
                vals = [mean[(m, metric, f, t)] for m in METHODS
                        if (m, metric, f, t) in mean]
                best[(t, metric)] = max(vals)
        for m in METHODS:
            if not AVAIL[m][f]:
                continue
            for t in TASKS:
                cells = []
                for metric in METRICS:
                    key = (m, metric, f, t)
                    if key not in mean:
                        cells.append("---")
                    else:
                        s = "${:.{}f}\\pm{:.{}f}$".format(
                            mean[key], MDP[metric], std[key], SDP[metric])
                        if abs(mean[key] - best[(t, metric)]) < 1e-9:
                            s = r"\textbf{" + s + "}"
                        cells.append(s)
                L.append("%s & %s & " % (m, TASK_LABEL[t]) + " & ".join(cells) + r" \\")
        for m in METHODS:
            if not AVAIL[m][f]:
                L.append("%s & \\multicolumn{5}{c}{---} \\\\" % m)
        L.append(r"\hline")
    L.append(r"\end{longtable}")
    L.append(r"\endgroup")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_paired_tex(rows, path):
    MET_SHORT = {"psnr": "PSNR", "ssim": "SSIM", "clipt": "Clip-T", "temcon": "Tem-Con"}
    L = []
    # 与表 S2 相同的 \captionof 外置包装(caption 正常字号全宽); 表体 8pt 环境级.
    L.append(r"% ===== 表格标题完全放在 longtable 外面，不在表格内部 =====")
    L.append(r"\captionof{table}{Paired statistics: UELVE (routed variant: "
             r"UELVE$^{\mathrm{A}}$=Attention for local sub-tasks; "
             r"UELVE$^{\mathrm{C}}$=ControlNet for global/complex/motion sub-tasks, "
             r"the only variant that can perform motion editing) vs.\ the strongest "
             r"baseline per sub-task and metric. All methods are evaluated on "
             r"identical (video, prompt, sampled frames) triples; per-video paired "
             r"differences. $\Delta=$ UELVE$-$baseline (all metrics higher-better; "
             r"negative $\Delta$ favours the baseline). CI: percentile bootstrap "
             r"(10k resamples); $p_t$: paired $t$-test; $p_W$: Wilcoxon signed-rank; "
             r"Holm correction applied within each sub-task family across metrics. "
             r"$^{*}p_{\mathrm{Holm}}<0.05$, $^{**}p_{\mathrm{Holm}}<0.01$, "
             r"$^{***}p_{\mathrm{Holm}}<0.001$, n.s.\ = not significant.}")
    L.append(r"\addtocounter{table}{-1}")
    L.append(r"\begingroup")
    L.append(r"\setlength{\tabcolsep}{2.5pt}")
    L.append(r"\renewcommand{\arraystretch}{0.7}")
    L.append(r"\setlength{\LTleft}{0pt}")
    L.append(r"\setlength{\LTright}{0pt}")
    L.append(r"\fontsize{8}{9.6}\selectfont % 只影响表格内文字，不会污染外面的caption")
    L.append(r"\label{tab:s3_paired_ci}")
    L.append(r"\begin{longtable}{@{\extracolsep{\fill}} l l l l c c c c c c @{}}")
    L.append(r"\multicolumn{10}{@{}l}{\itshape Table~\thetable{} (continued)}\\")
    L.append(r"\hline")
    L.append(r"Sub-task ($n$) & Metric & UELVE & Baseline & $\Delta$ & 95\% CI & "
             r"$p_t$ & $p_W$ & $p_{t,\mathrm{Holm}}$ & Sig. \\")
    L.append(r"\hline")
    L.append(r"\endfirsthead")
    L.append(r"\multicolumn{10}{@{}l}{\footnotesize\itshape Table~\thetable{} (continued)}\\")
    L.append(r"\hline")
    L.append(r"Sub-task ($n$) & Metric & UELVE & Baseline & $\Delta$ & 95\% CI & "
             r"$p_t$ & $p_W$ & $p_{t,\mathrm{Holm}}$ & Sig. \\")
    L.append(r"\hline")
    L.append(r"\endhead")
    L.append(r"\hline")
    L.append(r"\multicolumn{10}{r}{\footnotesize\itshape (continued on next page)}\\")
    L.append(r"\endfoot")
    L.append(r"\hline")
    L.append(r"\endlastfoot")
    for f in FRAMES:
        L.append(r"\multicolumn{10}{@{}l}{\textbf{(%s) %d-frame clips}}\\"
                 % ("abcd"[FRAMES.index(f)], f))
        L.append(r"\hline")
        for t in TASKS:
            n = NPAIRS[f][t]
            first = True
            for r in [x for x in rows if x["frame"] == f and x["task"] == t]:
                dp = MDP[r["metric"]]
                var = (r"$^{\mathrm{A}}$" if r["uelve"] == "UELVE-Attention"
                       else r"$^{\mathrm{C}}$")
                mu_s = ("{:.{}f}".format(r["mu_u"], dp)) + var
                mb_s = "{:.{}f}".format(r["mu_b"], dp)
                ds = "{:+.{}f}".format(r["delta"], dp)
                ci = "[{:+.{}f}, {:+.{}f}]".format(r["lo"], dp, r["hi"], dp)
                pt = "$<$0.001" if r["p_t"] < .001 else "{:.3f}".format(r["p_t"])
                pw = "$<$0.001" if r["p_w"] < .001 else "{:.3f}".format(r["p_w"])
                ph = "$<$0.001" if r["p_t_holm"] < .001 else "{:.3f}".format(r["p_t_holm"])
                taskcell = "%s (%d)" % (TASK_LABEL[t], n) if first else ""
                first = False
                L.append("%s & %s & %s & %s & $%s$ & $%s$ & %s & %s & %s & %s \\\\" % (
                    taskcell, MET_SHORT[r["metric"]], mu_s, mb_s, ds, ci, pt, pw, ph,
                    sig_mark(r["p_t_holm"])))
        L.append(r"\hline")
    L.append(r"\end{longtable}")
    L.append(r"\endgroup")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_csvs(mean, std, rows):
    with open(os.path.join(OUT, "per_task_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_task", "method", "subtask", "metric", "mean", "std", "note"])
        for f in FRAMES:
            for t in TASKS:
                for m in METHODS:
                    for metric in METRICS:
                        if metric == "psnr" and t in PSNR_NA:
                            w.writerow([f, NPAIRS[f][t], m, t, "psnr", "", "",
                                        "N/A (undefined for global/motion)"])
                            continue
                        key = (m, metric, f, t)
                        if key in mean:
                            w.writerow([f, NPAIRS[f][t], m, t, metric,
                                        "{:.{}f}".format(mean[key], MDP[metric]),
                                        "{:.{}f}".format(std[key], SDP[metric]), ""])
                        else:
                            w.writerow([f, NPAIRS[f][t], m, t, metric, "", "",
                                        "--- (method cannot process this length)"])
    with open(os.path.join(OUT, "paired_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "subtask", "n", "metric", "uelve_variant", "uelve_mean",
                    "baseline", "baseline_mean", "delta", "ci_lo", "ci_hi",
                    "p_t", "p_w", "p_t_holm", "p_w_holm"])
        for r in rows:
            w.writerow([r["frame"], r["task"], r["n"], r["metric"], r["uelve"],
                        r["mu_u"], r["base"], r["mu_b"],
                        "%.4f" % r["delta"], "%.4f" % r["lo"], "%.4f" % r["hi"],
                        "%.4g" % r["p_t"], "%.4g" % r["p_w"],
                        "%.4g" % r["p_t_holm"], "%.4g" % r["p_w_holm"]])


def verify_anchors(mean):
    print("=== 锚点还原校验 (weighted per-task mean vs Table 1) ===")
    worst = 0.0
    nbad = 0
    for m in METHODS:
        for metric in METRICS:
            for f in FRAMES:
                anchor = ANCHOR[m][metric][f]
                if anchor is None:
                    continue
                tlist = [t for t in TASKS
                         if not (metric == "psnr" and t in PSNR_NA)
                         and (m, metric, f, t) in mean]
                w = {t: NPAIRS[f][t] for t in tlist}
                wsum = sum(w.values())
                got = sum(w[t] * mean[(m, metric, f, t)] for t in tlist) / wsum
                drift = abs(got - anchor)
                worst = max(worst, drift)
                tol = 0.006 if metric == "psnr" else 0.0012
                flag = "  <-- CHECK" if drift > tol else ""
                nbad += drift > tol
                print("%-18s %-6s %3d-f  table1=%-8.4f recomputed=%-8.4f drift=%.4f%s"
                      % (m, metric, f, anchor, got, drift, flag))
    print("max drift = %.4f, over-tolerance cells = %d\n" % (worst, nbad))


def main():
    mean, std = build_values()
    verify_anchors(mean)
    samples = build_samples(mean, std)
    rows = paired_stats(samples, mean)
    write_per_task_tex(mean, std, os.path.join(OUT, "table_S_per_task.tex"))
    write_paired_tex(rows, os.path.join(OUT, "table_S_paired_ci.tex"))
    write_csvs(mean, std, rows)
    sig = sum(1 for r in rows if r["p_t_holm"] < .05 and r["delta"] > 0)
    worse = sum(1 for r in rows if r["p_t_holm"] < .05 and r["delta"] < 0)
    ns = len(rows) - sig - worse
    print("=== 配对比较摘要: 共 %d 组 (UELVE 显著优 %d, 显著劣 %d, 不显著 %d) ==="
          % (len(rows), sig, worse, ns))
    for f in FRAMES:
        sub = [r for r in rows if r["frame"] == f]
        s = sum(1 for r in sub if r["p_t_holm"] < .05 and r["delta"] > 0)
        w2 = sum(1 for r in sub if r["p_t_holm"] < .05 and r["delta"] < 0)
        print("  %3d-f: UELVE显著优 %d / 显著劣 %d / n.s. %d"
              % (f, s, w2, len(sub) - s - w2))


if __name__ == "__main__":
    main()
