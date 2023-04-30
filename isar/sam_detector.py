import numpy as np
import cv2
import os
import random

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

from params import OUTDIR
from util.isar_utils import performance_measure, semantic_obs_to_img, generate_pastel_color

from reidentification import Reidentification

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from segment_anything.utils.amg import batch_iterator, build_all_layer_point_grids, build_point_grid
from detector import Detector

#automatic mask generator:
from segment_anything.modeling import Sam
from segment_anything.utils.amg import MaskData, uncrop_boxes_xyxy, uncrop_points
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

from visualization_pca import VisualizationPca

from params import FeatureModes


class SAMDetector(Detector): # inherits from Detector, not really necessary just for consistency to make sure the same member functions are defined.

    def __init__(self, device = "cpu", sam_model_type = "vit_h", use_precomputed_embeddings = False, outdir = OUTDIR, n_per_side = 16, feature_mode = FeatureModes.CLIP_SAM) -> None:

        device = device  ## RIGHT now running on CPU, as Laptop GPU memory not sufficient
        model_type = sam_model_type
        sam_checkpoint = os.path.join('modelzoo', [i for i in os.listdir('modelzoo') if model_type in i][0])

        self.use_precomputed_embedding = use_precomputed_embeddings
        
        self.feature_mode = feature_mode

        self.start_reid = False

        self.show_images = True

        self.outdir = outdir

        self.point_grid = build_point_grid(n_per_side)

        # SAM model:
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.single_mask_generator = SingleCropMaskGenerator(self.predictor, points_per_side = n_per_side)

        self.template_feature = None

        self.clip = Reidentification()
        

        self.semantic_palette = np.array([generate_pastel_color() for _ in range(256)],dtype=np.uint8)

        self.visualize_pca = True
        
        if self.feature_mode == FeatureModes.SAM and self.visualize_pca:
            self.pca_vis = VisualizationPca()


    """
    gets called on the mouse callback of the UI
    """
    def on_click(self, x: int, y: int, img: np.ndarray, embedding: str):
        
        img_features = None

        x = int(x * 512 / img.shape[1])
        y = int(y * 512 / img.shape[0])
        sam_img = cv2.resize(img, (512, 512))

        # TODO: recompute Embeddings! (old ones seem to be corrupted because they were not computed on the resized image)
        self.set_img_embedding(sam_img, embedding)

        mask, iou_prediction, _ = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            box=None,
            multimask_output=False,
        )

        selected_prob = torch.tensor(iou_prediction)
        selected_box = self.get_bbox_from_mask(mask)
        freeze = sam_img
        cutout = self.get_cutout(sam_img, selected_box)

        show_mask = self.show_mask(mask)

        dst = cv2.addWeighted(sam_img.astype('uint8'), 0.7, show_mask.astype('uint8'), 0.3, 0).astype('uint8')
        cv2.imshow("seg", cv2.resize(dst, (img.shape[1], img.shape[0])))


        if self.feature_mode == FeatureModes.SAM or self.feature_mode == FeatureModes.CLIP_SAM:
            img_features = self.predictor.get_image_embedding().squeeze().cpu().numpy()
        
        self.template_feature = self.mask_to_features(sam_img, mask, img_features)


        if self.feature_mode == FeatureModes.SAM and self.visualize_pca:
            img_embedding_T = img_features
            img_embedding = img_embedding_T.transpose(1, 2, 0)
            img_embedding_flat = img_embedding.reshape(img_embedding.shape[0] * img_embedding.shape[1], img_embedding.shape[2])
            self.pca_vis.set_template(img_embedding_flat, mask, img_embedding.shape[0], img_embedding.shape[1])
            vis_embedding = self.pca_vis.transform(img_embedding_flat)
            self.pca_vis.visualize(vis_embedding, img.shape)
            cv2.waitKey(1)

        self.start_reid = True

        # TODO: save descriptor, check dimensions of all output variables and inputs to self.predictor.predict

        return cutout, show_mask, freeze, selected_prob, selected_box
    

    """
    gets called for every image
    """
    def detect(self, img: np.ndarray, image_name: str, embedding: str):
        scores = None
        boxes = None
        show_mask = None
        img_features = None

        if self.start_reid:

            sam_img = cv2.resize(img, (512, 512))
            # TODO: recompute Embeddings! (old ones seem to be corrupted because they were 
            # not computed on the resized image)
            self.set_img_embedding(sam_img, embedding)

            masks = [m for m in self.segment_all(sam_img, embedding) if np.sum(m) > 1000]

            boxes = []
            mask_descriptors = []
            with performance_measure("all object descriptors"):
                if self.feature_mode == FeatureModes.SAM or self.feature_mode == FeatureModes.CLIP_SAM:
                    img_features = self.predictor.get_image_embedding().squeeze().cpu().numpy()
                if self.feature_mode == FeatureModes.SAM and self.visualize_pca:
                    transposed_features = img_features.transpose(1, 2, 0)
                    img_embedding_flat = transposed_features.reshape(transposed_features.shape[0] * transposed_features.shape[1], transposed_features.shape[2])
                    vis_embedding = self.pca_vis.transform(img_embedding_flat)
                    self.pca_vis.visualize(vis_embedding, img.shape)
                    cv2.waitKey(1)

                for idx, mask in enumerate(masks):
                    mask_descriptor = self.mask_to_features(sam_img, mask, img_features)
                    mask_descriptors.append(mask_descriptor)
            
            n_masks = len(mask_descriptors)
            # remove nans
            mask_descriptor, masks = zip(*[(desc, msk) for desc, msk in zip(mask_descriptors, masks) if not np.isnan(desc).any()])

            n_nans_removed = n_masks - len(mask_descriptor)
            print("number of nans removed: ", n_nans_removed)

            mask_descriptors = np.array(mask_descriptors)

            # classifier 1: cosine similarity/dot product
            max_similarity, max_similarity_idx, similarities = self.compute_similarities_dotp(mask_descriptors)
            mask = masks[max_similarity_idx]

            # classifier 2: nearest neighbor
            nearest_neighbor_index, nearest_neighbor_descriptor, min_distance = self.get_nearest_neighbor_descriptor(mask_descriptors)
            # mask = masks[nearest_neighbor_index]

            print("classifiers match: ", max_similarity_idx == nearest_neighbor_index) 

            
            show_mask = self.show_mask(mask)
            show_mask = cv2.resize(show_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            dst = cv2.addWeighted(img.astype('uint8'), 0.7, show_mask.astype('uint8'), 0.3, 0).astype('uint8')

            print("maximum_similarity: ", max_similarity)
            print("min distance: ", min_distance)

            if self.show_images:
                cv2.imshow("seg", dst)

            cv2.imwrite(os.path.join(self.outdir, image_name), dst)

            self.predictor.reset_image()

            boxes = np.array(boxes)
            scores = np.array([similarities[i] for i in range(len(masks))])       

        return scores, boxes, show_mask

    def get_nearest_neighbor_descriptor(self, mask_descriptors):

        distances = distance.cdist(mask_descriptors, np.atleast_2d(self.template_feature))
        nearest_neighbor_index = np.argmin(distances)
        min_distance = distances[nearest_neighbor_index]
        nearest_neighbor_descriptor = mask_descriptors[nearest_neighbor_index]

        return nearest_neighbor_index, nearest_neighbor_descriptor, min_distance

    def compute_similarities_dotp(self, mask_descriptors):
        max_similarity = -np.inf
        similarities = []
        for idx, mask_descriptor in enumerate(mask_descriptors):
            similarity = np.dot(mask_descriptor, self.template_feature)
            similarities.append(similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_idx = idx
        
        return max_similarity, max_similarity_idx, similarities
    
    def mask_to_features(self, img, mask, img_features = None):
        """
        computes descriptor for a given mask and image. 
        If the feature mode is SAM, the image needs to be set before. (self.predictor.set_image(img))
        """
        if self.feature_mode == FeatureModes.SAM or self.feature_mode == FeatureModes.CLIP_SAM:
            transposed_features = img_features.transpose(1, 2, 0)
            sam_features = self.get_sam_object_descriptor(mask, img_features.shape, transposed_features)

        if self.feature_mode == FeatureModes.CLIP or self.feature_mode == FeatureModes.CLIP_SAM:
            bbox = self.get_bbox_from_mask(mask)
            clip_features = self.get_clip_features(img, mask, bbox)
        
        if self.feature_mode == FeatureModes.SAM:
            feature_vec = sam_features
        elif self.feature_mode == FeatureModes.CLIP:
            feature_vec = clip_features
        elif self.feature_mode == FeatureModes.CLIP_SAM:
            feature_vec = np.concatenate((sam_features, clip_features))
        
        feature_vec /= np.linalg.norm(feature_vec)
        return feature_vec

    def get_clip_features(self, img, mask, bbox):
        # mask image:
        mask = mask.squeeze()
        # set all pixels outside of mask to 0: #TODO: check how different colors/random image noise influences this.
        masked_image = img.copy()
        masked_image[mask==0,:] = np.array([255,255,255])
        masked_image = masked_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        # get clip features:
        clip_features = self.clip(masked_image)

        return clip_features.cpu().numpy()
    
    def segment_all(self, img: np.ndarray, embedding: str):

        with performance_measure("SAM mask generator"):

            res = self.single_mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            masks = [res[i]["segmentation"] for i in range(len(res))]

        return masks

    def set_img_embedding(self, image: np.ndarray, embedding = None):
        with performance_measure("SAM embedding"):

            # get image embedding:
            # precomputed_embeddings:
            if self.use_precomputed_embedding and embedding is not None:
                if os.path.exists(embedding):
                    input_image = self.predictor.transform.apply_image(image)
                    input_image_torch = torch.as_tensor(input_image, device=self.predictor.device)
                    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

                    assert (
                        len(input_image_torch.shape) == 4
                        and input_image_torch.shape[1] == 3
                        and max(*input_image_torch.shape[2:]) == self.predictor.model.image_encoder.img_size
                    ), f"set_torch_image input must be BCHW with long side {self.predictor.model.image_encoder.img_size}."
                    self.predictor.reset_image()

                    self.predictor.original_size = image.shape[:2]
                    self.predictor.input_size = tuple(input_image_torch.shape[-2:])
                    self.predictor.features = torch.load(embedding, map_location=self.predictor.device)
                    self.predictor.is_image_set = True
                
                # no precomputed embeddings available:
                else:
                    self.predictor.set_image(image, image_format='BGR')

                    if not os.path.exists(os.path.split(embedding)[0]):
                        os.makedirs(os.path.split(embedding)[0])

                    features = self.predictor.get_image_embedding()
                    torch.save(features, embedding)

            else:
                self.predictor.set_image(image, image_format='BGR')

    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image
    
    def get_sam_object_descriptor(self, mask: np.ndarray, features_shape, transposed_features: np.ndarray):

        mask_sml = cv2.resize(mask.squeeze().astype(float), (features_shape[2], features_shape[1])) > 0.5

        mask_feature_vectors = transposed_features[mask_sml]
        # mask_feature_vectors_norm = mask_feature_vectors / np.linalg.norm(mask_feature_vectors, axis=1, keepdims=True)
        mask_feature_vector_mean = np.mean(mask_feature_vectors, axis=0)

        mask_feature_vector_mean = mask_feature_vector_mean / np.linalg.norm(mask_feature_vector_mean)

        return mask_feature_vector_mean
    
    def get_bbox_from_mask(self, mask: np.ndarray):
        mask = mask.squeeze().astype(np.uint8)
        
        row_indices, col_indices = np.where(mask)

        # Calculate the min and max row and column indices
        row_min, row_max = np.min(row_indices), np.max(row_indices)
        col_min, col_max = np.min(col_indices), np.max(col_indices)

        # Return the bounding box coordinates as a tuple
        return (col_min, row_min, col_max, row_max)
    
    def combine_masks(self, masks, img_size):
        combined_mask = np.zeros(img_size, dtype=np.uint32)
        num_masks = len(masks)

        contours = []

        for i, mask in enumerate(masks):
            combined_mask[mask] = i + 1

            contours.append(cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])

        return combined_mask
    

"""
    SingleCropMaskGenerator
    
    This class inherits from the SamAutomaticMaskGenerator and is used to generate panoptic masks for a given image.
    The difference to the SamAutomaticMaskGenerator is that it always only uses a single crop (the entire image) to compute backbone on.
    That way we only have to compute the backbone once per image.

"""
class SingleCropMaskGenerator(SamAutomaticMaskGenerator):

    # Default values are taken from the SamAutomaticMaskGenerator
    def __init__(self, predictor: SamPredictor, points_per_side = 32) -> None:
        super().__init__(predictor.model, points_per_side = points_per_side, points_per_batch = 64, pred_iou_thresh = 0.88, 
                         stability_score_thresh = 0.95, stability_score_offset = 1.0, 
                         box_nms_thresh = 0.7, crop_n_layers= 0, crop_nms_thresh = 0.7, 
                         crop_overlap_ratio= 512 / 1500, crop_n_points_downscale_factor = 1, 
                         point_grids = None, min_mask_region_area = 1000, output_mode = "binary_mask")
        
        self.crop_n_layers = 0
        self.predictor = predictor
    

    """
    redefined _process_crop, such that set_image() is not called again, but the precomputed embedding is used
    """
    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        cropped_im = image
        cropped_im_size = cropped_im.shape[:2]

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"])),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data



from params import DATADIR, DATASET, EVALDIR, IMGDIR, TASK, OUTDIR, FPS, EMBDIR
from util.isar_utils import get_image_it_from_folder


"""
used for debug:
- generates panoptic masks.
"""
def main():
    if not os.path.exists(os.path.join("/home/nico/semesterproject/test/", TASK)):
        os.makedirs(os.path.join("/home/nico/semesterproject/test/", TASK))
                
    detector = SAMDetector("cpu", "vit_h")
    single_mask_generator = SingleCropMaskGenerator(detector.predictor, points_per_side = 16)
    detector.use_precomputed_embedding = True
    
    images = get_image_it_from_folder(IMGDIR)


    cv2.namedWindow("mask")

    while True:
        image = images.__next__()
        embedding = os.path.join(EMBDIR, image.replace(".jpg", ".pt"))

        img = cv2.imread(os.path.join(IMGDIR, image))

        detector.set_img_embedding(img, embedding)

        with performance_measure("automatic mask generation"):
            res = single_mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            masks = [res[i]["segmentation"] for i in range(len(res))]


        combined_mask = detector.combine_masks(masks, img.shape[:2])
        mask_img = semantic_obs_to_img(combined_mask, detector.semantic_palette)
        overlayed_mask_image = cv2.addWeighted(img.astype('uint8'), 0.2, mask_img.astype('uint8'), 0.8, 0).astype('uint8')

        cv2.imshow("mask", overlayed_mask_image)


        k = cv2.waitKey(1) #& 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()