# Precise segmentation masks via GroundingDINO-Tiny (text->box) + FastSAM (box->mask).
#
# Ported from Video-P2P (gen_masks.py) for ELVE. Used by test_fatezero.py to build an
# external `fixed_mask` for SpatialBlender, replacing the cross-attention-derived mask.
#
# Typical use in ELVE:
#     from mask_utils import gen_masks
#     gd_proc, gd_model, fsam = gen_masks.load_mask_models(device, mask_model_path)
#     mask = gen_masks.generate_word_mask(frames_np, word, gd_proc, gd_model, fsam,
#                                         device, box_threshold, dilate_px)   # [F,512,512] bool
#
# Robustness: if a word is not detected in a frame, the box threshold is stepped down; if
# still nothing, the previous frame's mask is reused.

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
from PIL import Image
import torch
import cv2

# Default weights live in the Video-P2P repo (shared on this machine).
DEFAULT_MASK_MODEL_PATH = "/mnt/fspace/hproject/Video-P2P-main/pretrained/mask_models"


def load_512_seq(image_path, n=8):
    """Load the first n frames from a directory as 512x512 center-cropped uint8 images.

    Kept for standalone smoke testing; the edit pipeline instead feeds the already-loaded,
    already-aligned `batch["images"]` frames directly into generate_word_mask.
    """
    files = sorted(os.listdir(image_path))
    frames = []
    for i in range(n):
        im = np.array(Image.open(os.path.join(image_path, files[i])).convert("RGB"))
        h, w, c = im.shape
        if h < w:
            off = (w - h) // 2; im = im[:, off:off + h]
        elif w < h:
            off = (h - w) // 2; im = im[off:off + w]
        im = np.array(Image.fromarray(im).resize((512, 512)))
        frames.append(im)
    return np.stack(frames)


def load_mask_models(device, mask_model_path=DEFAULT_MASK_MODEL_PATH):
    """Load GroundingDINO-Tiny + FastSAM once. Returns (gd_proc, gd_model, fsam)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from ultralytics import FastSAM
    gd_path = os.path.join(mask_model_path, "grounding-dino-tiny")
    fsam_path = os.path.join(mask_model_path, "FastSAM-x.pt")
    gd_proc = AutoProcessor.from_pretrained(gd_path)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_path).to(device).eval()
    fsam = FastSAM(fsam_path)
    return gd_proc, gd_model, fsam


def detect_boxes(word, image_np, thr, gd_proc, gd_model, device):
    pil = Image.fromarray(image_np)
    text = word.lower().strip(); text = text + ("." if not text.endswith(".") else "")
    inputs = gd_proc(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gd_model(**inputs)
    res = gd_proc.post_process_grounded_object_detection(
        out, inputs.input_ids, box_threshold=thr, text_threshold=max(thr - 0.05, 0.1),
        target_sizes=[pil.size[::-1]])
    boxes = []
    for b, s, l in zip(res[0]["boxes"], res[0]["scores"], res[0]["labels"]):
        if l.strip().lower() == word.lower():
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)))
    return boxes


def seg_in_box(image_np, box, fsam, device):
    # FastSAM's `device` arg wants a GPU int. Accept "cuda", "cuda:1", or torch.device.
    _dev = torch.device(device)
    _gpu = _dev.index if _dev.index is not None else 0
    r = fsam(image_np, device=_gpu, retina_masks=True, imgsz=512,
             conf=0.4, iou=0.9, verbose=False)[0]
    masks = r.masks.data
    if masks is None or len(masks) == 0:
        return None
    x1, y1, x2, y2 = box
    H, W = masks.shape[1], masks.shape[2]
    bx = torch.zeros((H, W), device=masks.device); bx[int(y1):int(y2), int(x1):int(x2)] = 1
    inter = (masks.bool() & bx.bool()).float().sum((1, 2))
    union = (masks.bool() | bx.bool()).float().sum((1, 2))
    iou = inter / union.clamp(min=1)
    return masks[int(iou.argmax())].cpu().numpy().astype(bool)


def _threshold_cascade(box_threshold):
    """Start at the requested threshold and step DOWN for robustness (floor 0.10)."""
    thrs = [box_threshold, box_threshold - 0.10, box_threshold - 0.20]
    thrs = [round(t, 3) for t in thrs if t >= 0.10]
    return sorted(set(thrs), reverse=True)


def generate_word_mask(frames, word, gd_proc, gd_model, fsam, device,
                       box_threshold=0.20, dilate_px=4, verbose=False):
    """Return [F,512,512] bool mask for `word` across `frames` ([F,512,512,3] uint8).

    Per frame: detect at box_threshold, stepping down if nothing is found; reuse the
    previous frame's mask only as a last resort.
      - box_threshold LOWER  -> more lenient detection (catches the object in more frames).
      - dilate_px HIGHER     -> spatially expands the mask (the direct "range" knob).
    """
    n = frames.shape[0]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    masks = np.zeros((n, 512, 512), dtype=bool)
    thrs = _threshold_cascade(box_threshold)
    prev = None
    for fi in range(n):
        f = frames[fi]
        box, used_thr = None, None
        for thr in thrs:
            bs = detect_boxes(word, f, thr, gd_proc, gd_model, device)
            if bs:
                box = max(bs, key=lambda b: b[4])[:4]; used_thr = thr; break
        m = seg_in_box(f, box, fsam, device) if box is not None else None
        if m is None or m.sum() < 50:
            m = prev if prev is not None else np.zeros((512, 512), bool)
            tag = "reuse"
        else:
            m = cv2.dilate(m.astype(np.uint8), kern).astype(bool)
            tag = "fresh"
        prev = m
        masks[fi] = m
        if verbose:
            print(f"  [maskgen] {word} frame{fi}: {tag} (thr={used_thr}, mask_px={int(m.sum())}, "
                  f"fill={100 * m.sum() / 262144:.1f}%)")
    return masks


if __name__ == "__main__":
    # Standalone smoke test: python -m mask_utils.gen_masks --image-path data/man-skicing8f --words man
    import argparse
    from PIL import ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-path", default="data/man-skicing8f")
    ap.add_argument("--words", default="man")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--dilate-px", type=int, default=4)
    ap.add_argument("--mask-model-path", default=DEFAULT_MASK_MODEL_PATH)
    a = ap.parse_args()

    device = f"cuda:{a.gpu}"
    frames = load_512_seq(a.image_path, n=a.n_frames)
    gd_proc, gd_model, fsam = load_mask_models(device, a.mask_model_path)
    for word in a.words.split(","):
        m = generate_word_mask(frames, word, gd_proc, gd_model, fsam, device,
                               box_threshold=a.box_threshold, dilate_px=a.dilate_px, verbose=True)
        sheet = Image.new("RGB", (512 * a.n_frames, 512), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for fi in range(a.n_frames):
            arr = frames[fi].copy(); tint = np.array([255, 0, 0], np.uint8)
            arr[m[fi]] = (0.5 * arr[m[fi]] + 0.5 * tint).astype(np.uint8)
            sheet.paste(Image.fromarray(arr), (fi * 512, 0))
            d.text((fi * 512 + 4, 2), f"{word} f{fi}", fill=(255, 255, 0))
        out = f"_masktest_{word}.png"
        sheet.save(out)
        print(f"[maskgen] {word}: saved contact sheet -> {out}, mean fill={m.mean():.3f}")
