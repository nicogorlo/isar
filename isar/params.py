import os
## general: 

FPS: int = 15


## OW-DETR:

KEEP_LOGITS = 0.7
KEEP_OBJECTNESS = 0.0
NUM_BOXES = 100
ADA_THRESH_MAX_OVERLAP = 0.4
ADA_THRESH_MULTIPLIER = 1.05



## Data: (not relevant when running benchmark)

# #AR-KIT:
# DATASET = 'AR-KIT'
# TASK = "47670305"
# DATADIR = "/home/nico/semesterproject/data/ARKit_data/3dod/Training/"
# IMG_DIR = f'{TASK}/{TASK}_frames/lowres_wide/'
# EVAL_DIR = None

# #Cityscapes:
# DATASET = 'Cityscapes'
# TASK = 'stuttgart_00'
# DATADIR = "/home/nico/semesterproject/data/Cityscapes/"
# IMG_DIR = os.path.join('leftImg8bit_demoVideo/leftImg8bit/demoVideo/', TASK)
# EVAL_DIR = None

# Habitat single object tracking:

# DATASET = 'Habitat_single_obj'
# TASK = 'apartment_1_loiter_bike'
# DATADIR = "/home/nico/semesterproject/data/habitat_single_object_tracking/"
# IMGDIR = os.path.join(DATADIR, TASK, "rgb")
# EVALDIR = os.path.join(DATADIR, TASK, "semantics")
# EMBDIR = os.path.join(DATADIR, TASK, "embeddings")
# OUTDIR = os.path.join("/home/nico/semesterproject/test/", TASK)

# DAVIS:
DATASET = 'DAVIS'
TASK = "bear"
DATADIR = "/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/"
IMGDIR = os.path.join(DATADIR, "JPEGImages/480p/", TASK)
EVALDIR = os.path.join(DATADIR, "Annotations_unsupervised/480p/", TASK)
EMBDIR = os.path.join(DATADIR, "ImageEmbeddings/", TASK)
OUTDIR = os.path.join("/home/nico/semesterproject/test/", TASK)