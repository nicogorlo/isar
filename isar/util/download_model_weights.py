import gdown
import requests
import os

dirname = os.path.dirname(__file__)
if not os.path.exists(os.path.join(dirname, '../modelzoo')):
    os.makedirs(os.path.join(dirname, '../modelzoo'))

if not os.path.exists(os.path.join(dirname, '../feature_dict_dinov2')):
    os.makedirs(os.path.join(dirname, '../feature_dict_dinov2'))

r = requests.get('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', allow_redirects=True)
open(os.path.join(dirname, '../modelzoo/sam_vit_h_4b8939.pth'), 'wb').write(r.content)


gdown.download('https://drive.google.com/uc?id=15nlfT0DwwEBQwNzS_gxs5SwMZwDf0lLE&export=download', 
               os.path.join(dirname, '../feature_dict_dinov2/dino_features_dinov2_vitl14.pkl'), quiet=False)