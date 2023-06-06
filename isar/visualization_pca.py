import numpy as np
import cv2
import os
import argparse
import json

import torch
import torchvision.transforms as T
from PIL import Image

from sklearn.decomposition import PCA
from segment_anything import SamPredictor, sam_model_registry

class VisualizationPca():
    
    def __init__(self):
        cv2.namedWindow('pca_dino')
        cv2.namedWindow('pca_sam')
        self.upsampler = torch.nn.Upsample(size=(512, 512), mode='bicubic', align_corners=False)
        torch.set_grad_enabled(False)

        self.patch_h, self.patch_w = 40, 40
        
    def initialize_sam(self, device):
        sam_model_type = 'vit_h'
        sam_checkpoint = os.path.join('modelzoo', [i for i in os.listdir('modelzoo') if sam_model_type in i][0])        
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor

    def initialize_dino(self):
        dino_model_type = 'dinov2_vitl14'
        model = torch.hub.load('facebookresearch/dinov2', dino_model_type)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    
    def extract_descriptors_sam(self, sam_predictor, image):
        img_resized = cv2.resize(image, (512, 512))
        sam_predictor.set_image(img_resized, image_format='BGR')
        img_features = sam_predictor.get_image_embedding().squeeze().cpu()
        img_embedding = img_features.permute((1, 2, 0))

        return img_embedding
    
    def extract_descriptors_dino(self, dino_predictor, image):
        transform = T.Compose([
        T.Resize((self.patch_h * 14, self.patch_w * 14)),
        T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).cuda()
        features = dino_predictor.forward_features(input_tensor)['x_norm_patchtokens'].squeeze().cpu()
        features = features.reshape(self.patch_h, self.patch_w, features.shape[-1])
        return features

    def set_template(self, descriptors: np.ndarray, mask: torch.Tensor):
    
        self.template_descriptors = descriptors[mask, :]
        self.pca = PCA(n_components=3)
        self.pca.fit(self.template_descriptors)

    def transform(self, descriptors: np.ndarray):
        principal_components = self.pca.transform(descriptors)
        # min max normalization:
        principal_components = self.min_max_normalization(principal_components)

        return principal_components
    
    def visualize(self, principal_components: np.ndarray, img_shape):
        principal_components = principal_components
        principal_components = principal_components.reshape(img_shape)
        cv2.imshow("pca", principal_components.astype("float64"))

    def min_max_normalization(self,arr):
        # Perform min-max normalization for each channel
        min_vals = np.min(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        normalized_arr = (arr - min_vals) / (max_vals - min_vals)

        return normalized_arr
    
    def get_data(self, datadir, imgname_train, imgname_test, semantic_id):
        img_train = cv2.imread(os.path.join(datadir, 'color', imgname_train))
        img_test = cv2.imread(os.path.join(datadir, 'color', imgname_test))

        mask_train = cv2.imread(os.path.join(datadir, 'semantic', imgname_train.replace("jpg", "png")))

        with open(os.path.join(datadir, 'color_map.json'), 'r') as f:
            color_map = json.load(f)

        mask_train = np.all(mask_train == color_map[str(semantic_id)], axis=-1)

        return img_train, img_test, mask_train

    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image.astype("uint8")

def main(data_dir, out_dir, imgname_train, imgname_test, semantic_id, device):
    # initialize sam
    pca_vis = VisualizationPca()
    sam_predictor = VisualizationPca().initialize_sam(device)
    dino_predictor = VisualizationPca().initialize_dino()

    img_train, img_test, mask_train = pca_vis.get_data(data_dir, imgname_train, imgname_test, semantic_id)

    img_annotation = cv2.addWeighted(img_train, 0.7, pca_vis.show_mask(mask_train), 0.3, 0)

    cv2.imwrite(os.path.join(out_dir,"pca_annotation.png"), img_annotation[9*20:-9*20, 16*20:-16*20, :])

    pca_vis.upsampler = torch.nn.Upsample(size=(img_train.shape[0], img_train.shape[1]), mode='bicubic', align_corners=False)

    descriptors_sam = pca_vis.extract_descriptors_sam(sam_predictor, img_train)
    upsampled_sam = pca_vis.upsampler(descriptors_sam.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    pca_vis.set_template(upsampled_sam, mask_train)

    descriptors_sam = pca_vis.extract_descriptors_sam(sam_predictor, img_test)
    upsampled_sam = pca_vis.upsampler(descriptors_sam.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()

    pca_img = pca_vis.transform(upsampled_sam.reshape(img_train.shape[0]*img_train.shape[1], upsampled_sam.shape[-1]))
    pca_img = pca_img.reshape(img_train.shape[0], img_train.shape[1], pca_img.shape[-1]).astype("float32")
    cv2.imshow("pca_sam", pca_img)

    #save cropped image
    cv2.imwrite(os.path.join(out_dir,"pca_sam.png"), (pca_img[9*20:-9*20, 16*20:-16*20, :] * 255).astype("uint8"))
    cv2.imwrite(os.path.join(out_dir,"pca_inference.png"), img_test[9*20:-9*20, 16*20:-16*20, :])


    descriptors_dino = pca_vis.extract_descriptors_dino(dino_predictor, img_train)
    upsampled_dino = pca_vis.upsampler(descriptors_dino.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    pca_vis.set_template(upsampled_dino, mask_train)

    descriptors_dino = pca_vis.extract_descriptors_dino(dino_predictor, img_test)
    upsampled_dino = pca_vis.upsampler(descriptors_dino.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()

    pca_img = pca_vis.transform(upsampled_dino.reshape(img_train.shape[0]*img_train.shape[1], upsampled_dino.shape[-1]))
    pca_img = pca_img.reshape(img_train.shape[0], img_train.shape[1], pca_img.shape[-1]).astype("float32")
    cv2.imshow("pca_dino", pca_img)

    cv2.imwrite(os.path.join(out_dir,"pca_dino.png"), (pca_img[9*20:-9*20, 16*20:-16*20, :] * 255).astype("uint8"))

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualize PCA')

    parser.add_argument('--device', type=str, default='cpu', help='device')

    parser.add_argument('--datadir', type=str, default='/home/nico/semesterproject/data/re-id_benchmark_ycb/single_object/cans/train/cans_on_counter/', help='path to data')

    parser.add_argument('--outdir', type=str, default='/home/nico/semesterproject/test/screenshots_debug/PCA', help='path to output directory')

    parser.add_argument('--img_train', type=str, default='0000083.jpg', help='path to image')

    parser.add_argument('--img_test', type=str, default='0000343.jpg', help='path to image')

    parser.add_argument('--semantic_id', type=int, default=1100, help='semantic id')

    args = parser.parse_args()

    main(args.datadir, args.outdir, args.img_train, args.img_test, args.semantic_id, args.device)
