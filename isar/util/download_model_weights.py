import gdown
import requests
import os

if not os.path.exists('../modelzoo'):
    os.makedirs('../modelzoo')

if not os.path.exists('../feature_dict_dinov2'):
    os.makedirs('../feature_dict_dinov2')

r = requests.get('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', allow_redirects=True)
open('../modelzoo/sam_vit_h_4b8939.pth', 'wb').write(r.content)


gdown.download('https://drive.google.com/uc?id=15nlfT0DwwEBQwNzS_gxs5SwMZwDf0lLE&export=download', 
               '../feature_dict_dinov2/dino_features_dinov2_vitl14.pkl', quiet=False)