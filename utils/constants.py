
import os
from datetime import datetime 

date_time = datetime.now().strftime("%m-%d-%Y")

CWD = os.getcwd()
MP4_PATH = os.path.join(CWD, 'data', 'mp4_videos')
OUTPUT_PATH = os.path.join(CWD, 'results', date_time) 

GENERATED_DATA_PATH = os.path.join(CWD, 'generated', 'data')
PREPROCESSOR_DICT = {
    'lineart_realistic': "/mnt/fspace/hproject/model/control_v11p_sd15_lineart/",
    'lineart_coarse': "/mnt/fspace/hproject/model/control_v11p_sd15_lineart/",
    'lineart_standard': "/mnt/fspace/hproject/model/control_v11p_sd15_lineart/",
    'lineart_anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'lineart_anime_denoise': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'softedge_hed': '/mnt/fspace/hproject/model/control_v11p_sd15_softedge/',
    'softedge_hedsafe': '/mnt/fspace/hproject/model/control_v11p_sd15_softedge/',
    'softedge_pidinet': '/mnt/fspace/hproject/model/control_v11p_sd15_softedge/',
    'softedge_pidsafe': '/mnt/fspace/hproject/model/control_v11p_sd15_softedge/',
    'canny': '/mnt/fspace/hproject/model/control_v11p_sd15_canny',
    'depth_leres': '/mnt/fspace/hproject/model/control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87/',
    'depth_leres++': '/mnt/fspace/hproject/model/control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87/',
    'depth_midas': '/mnt/fspace/hproject/model/control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87/',
    'depth_zoe': '/mnt/fspace/hproject/model/control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87/',
}

MODEL_IDS = {
    'Realistic Vision V5.1': '130072',
    'Realistic Vision V6.0' : '245598',
    'MajicMIXRealisticV7' : '176425',
    'DreamShaper' : '128713',
    'EpicPhotoGasm' : '223670',
    'DivineEleganceMix (Anime)': '238656',
    'GhostMix (Anime)': '76907',
    'CetusMix (Anime)': '105924',
    'Counterfeit (Anime)': '57618',
    'SD 1.5': 'None'
}

