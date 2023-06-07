import numpy as np
import os
from PIL import Image
from typing import List, Tuple
import cv2
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim

from segment_anything import SamPredictor, sam_model_registry
from detector import GenericDetector

from util.isar_utils import performance_measure, semantic_obs_to_img, generate_pastel_color
from sam_mask_generator import SingleCropMaskGenerator

class BaselineMethod(GenericDetector):
    def __init__(self, device: str, sam_model_type: str = 'vit_h', dino_model_type: str = 'dinov2_vitl14', use_precomputed_sam_embeddings: bool = False, outdir: str = '', *args, **kwargs):

        #configurable parameters:
        self.start_reid = False
        self.show_images = True
        self.device = device
        self.outdir = outdir

        # SAM model:
        sam_checkpoint = os.path.join('modelzoo', [i for i in os.listdir('modelzoo') if sam_model_type in i][0])
        self.use_precomputed_sam_embedding = use_precomputed_sam_embeddings
        
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

        # DINO model:
        self.dino_model = self.load_dino_v2_model(dino_model_type).to(device='cuda')

        self.patch_h, self.patch_w = 40, 40
        self.upsampled_h, self.upsampled_w = 512, 512

        self.upsampled_feature_vectors = True # TODO: delete all occurences
        self.use_conv = False
        self.use_sam_refinement = True

        self.upsampler = nn.Upsample(size=(self.upsampled_h, self.upsampled_w), mode='bicubic', align_corners=False)

        # classifier:
        self.reg = 0.00
        self.loss = lambda out, label, weights: (torch.mean(torch.clamp(1 - label * out, min=0))
                                                 + self.reg * (weights.t() @ weights) / 2.0)

        if self.use_conv:
            self.svm_conv = nn.Conv1d(self.dino_model.num_features, 1, kernel_size=1, stride=1, padding=0, device='cuda')
        else:
            self.svm = nn.Linear(self.dino_model.num_features, 1).to(device='cuda')

        self.svms = {}
        self.n_negative_features_same_image = 512*512

        with open(f"feature_dict_dinov2/dino_features_{dino_model_type}.pkl", "rb") as f:
            self.negative_sample_dict = pickle.load(f)

        self.margins = {}

        self.color_map = {}

    def on_new_task(self, info: dict):
        super().on_new_task()

        self.info = info
        self.color_map = info["color_map"]
        self.svms = {}
        self.margins = {}
    
    def train(
            self,
            train_dir: str,
            train_scenes: list,
            semantic_ids: list,
            prompts: dict
            ):
        prompts_rearranged: dict = {}
        for scene in train_scenes:
            scene_prompts = {}
            prompt_dict = prompts[scene]
            for semantic_id in semantic_ids:
                scene_prompts[semantic_id] = {}
            for img, prompt in prompt_dict.items():
                for semantic_id in prompt.keys():
                    scene_prompts[int(semantic_id)].update({os.path.join(train_dir, scene, "color/", img + ".jpg"): prompt[semantic_id]})
            prompts_rearranged.update(scene_prompts)
        

        ###################################################
        #### TODO: create SVM for each semantic class: ####
        ###################################################
        
        positive_samples = {}
        for semantic_id in semantic_ids:
            prompts_class = prompts_rearranged[semantic_id]
            imgs = []
            embeddings_sam = []
            for image_path in list(prompts_class.keys()):
                imgs.append(cv2.imread(image_path))
                embeddings_sam.append(image_path.replace("color", "embeddings").replace(".jpg", ".pt"))

            self.start_reid = True
            torch.set_grad_enabled(True)

            flat_masks_list = []
            img_features_list = []

            for idx, img in enumerate(imgs):
                # TODO: could add mirrored image for initial training

                img_square = cv2.resize(img, (self.upsampled_h, self.upsampled_w), interpolation=cv2.INTER_LINEAR)
                # 1. predict mask with sam_predictor
                mask = self.initial_sam_prediction(0, 0, img_square, img, embeddings_sam[idx], None, list(prompts_class.values())[idx]) 

                cv2.imshow("seg", 
                           cv2.resize(cv2.addWeighted(img_square.astype("uint8"), 0.3, self.show_mask(mask).astype("uint8"), 0.7, 0), (img.shape[1],img.shape[0])))
                cv2.waitKey(1)
                
                # cv2.waitKey(100)
                # 2. preprocess image dino
                tensor_in = self.preprocess_image_dino(img)

                # 3. compute masked dino embeddings
                img_features, mask_flat = self.compute_img_dino_embeddings(tensor_in, mask)

                flat_masks_list.append(mask_flat)
                img_features_list.append(img_features)
            img_features = torch.cat(img_features_list, dim=0)
            mask_flat = np.concatenate(flat_masks_list, axis=0)

            task = train_dir.split("/")[-3]
            tensor_dataset, weights = self.create_svm_dataset(img_features, mask_flat, "Habitat_single_obj", task)            
            num_samples = len(tensor_dataset)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
            dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=2048, sampler = sampler, num_workers=0)

            if self.use_conv:
                svm_conv = nn.Conv1d(self.dino_model.num_features, 1, kernel_size=1, stride=1, padding=0, device='cuda')
                svm_conv, loss_list = self.train_svm_conv(svm_conv, dataloader)
                self.svms[semantic_id] = svm_conv
                self.margins[semantic_id] = 2.0 / np.linalg.norm(svm_conv.weight.detach().cpu().numpy())
            else:
                svm = nn.Linear(self.dino_model.num_features, 1).to(device='cuda')
                svm, loss_list = self.train_svm(svm, dataloader)
                self.svms[semantic_id] = svm
                self.margins[semantic_id] = 2.0 / np.linalg.norm(svm.weight.detach().cpu().numpy())

            del tensor_in, tensor_dataset, sampler, dataloader
            torch.cuda.empty_cache()
            torch.set_grad_enabled(False)
        
        return

    def test(self, img: np.ndarray, image_name: str, embedding: str) -> np.ndarray:
        show_mask = None

        if self.start_reid:
            tensor = self.preprocess_image_dino(img).cuda()
            img_features = self.extract_features_dino(self.dino_model, tensor).cuda()
            del tensor
            torch.cuda.empty_cache()

            img_square = cv2.resize(img, (self.upsampled_w, self.upsampled_h))
            self.set_sam_img_embedding(img_square, embedding)

            mask_info = {}

            for semantic_id in sorted(list(self.svms.keys())):
                out, svm_predictions = self.predict_svm(img_features, img, semantic_id)
                mask_info[semantic_id] = (out, svm_predictions)# / self.margins[semantic_id])

            combined_mask = self.predict_multi(img_features, img_square)
            # cv2.imshow("mask_comb1100", cv2.addWeighted(img_square.astype("uint8"), 0.3, self.show_mask(mask_info[1100][0]).astype("uint8"), 0.7, 0.0))
            # cv2.waitKey(1)
            # cv2.imshow("mask_comb1103", cv2.addWeighted(img_square.astype("uint8"), 0.3, self.show_mask(mask_info[1103][0], color = np.array([0, 255, 0])).astype("uint8"), 0.7, 0.0))
            # cv2.waitKey(1)
            # combined_mask = self.combine_masks(mask_info)

            combined_mask_rgb = np.zeros((self.upsampled_h, self.upsampled_w, 3))
            for semantic_id in sorted(list(self.svms.keys())): 
                mask = (combined_mask == semantic_id)
                mask_colored = self.show_mask(mask.astype("float32"),np.array(self.color_map[str(semantic_id)]))
                combined_mask_rgb[mask] = mask_colored[mask]

            combined_mask_rgb = cv2.resize(combined_mask_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype("uint8")
            dst = cv2.addWeighted(img, 0.7, combined_mask_rgb, 0.3, 0)

            if self.show_images:
                cv2.imshow("seg", dst)

            cv2.waitKey(int(1000/30))
            cv2.imwrite(os.path.join(self.outdir, image_name), dst)

            self.sam_predictor.reset_image()
            del img_features, svm_predictions, out
            torch.cuda.empty_cache()

        return combined_mask_rgb
    
    def predict_svm(self, img_features: torch.Tensor, img: np.ndarray, semantic_id: int):
        if self.use_conv:
            svm_predictions = self.svms[semantic_id](img_features.unsqueeze(-1)).detach().cpu().numpy()
        else: 
            svm_predictions = self.svms[semantic_id](img_features).detach().cpu().numpy()
        
        self.visualize_svm_distance(svm_predictions, img, semantic_id)
        
        out = (svm_predictions > self.margins[semantic_id]/2).reshape((self.upsampled_h, self.upsampled_w))

        disconnected_masks = self.one_hot_encode_binary_mask(out)
        if disconnected_masks:
            filtered_mask = np.logical_or.reduce(disconnected_masks)
            out = self.sam_refinement(img, filtered_mask, svm_predictions.reshape((self.upsampled_h, self.upsampled_w)))
        else:
            out = np.zeros_like(out, bool)

        return out, svm_predictions.reshape((self.upsampled_h, self.upsampled_w))

    def predict_multi(self, img_features: torch.Tensor, img_square: np.ndarray):

        def calculate_reward(prediction_array, mask):
            return (prediction_array * mask).sum()
        
        def create_reward_matrix(predictions_dict, masks):
            reward_matrix = []
            for semantic_id, prediction_array in predictions_dict.items():
                reward_row = []
                for mask in masks:
                    reward_row.append(calculate_reward(prediction_array, mask))
                reward_matrix.append(reward_row)
                # print(semantic_id, "|", reward_row)
                # print("----------------------------------------")
            return reward_matrix
        
        def assign_masks(predictions_dict, masks):
            reward_matrix = create_reward_matrix(predictions_dict, masks)
            row_ind, col_ind = linear_sum_assignment(reward_matrix, maximize = True)
            return ({list(predictions_dict.keys())[i]: masks[j] for i, j in zip(row_ind, col_ind)},
                    {list(predictions_dict.keys())[i]: reward_matrix[i][j] for i, j in zip(row_ind, col_ind)})

        svm_predictions = {}
        out = {}
        for semantic_id in sorted(list(self.svms.keys())):
            if self.use_conv:
                svm_predictions[semantic_id] = self.svms[semantic_id](img_features.unsqueeze(-1)).detach().cpu().numpy()
            else: 
                svm_predictions[semantic_id] = self.svms[semantic_id](img_features).detach().cpu().numpy()

            out[semantic_id] = (svm_predictions[semantic_id] > self.margins[semantic_id]/2).reshape((self.upsampled_h, self.upsampled_w))
            svm_predictions[semantic_id] = svm_predictions[semantic_id].reshape((self.upsampled_h, self.upsampled_w))

            disconnected_masks = self.one_hot_encode_binary_mask(out[semantic_id])
            if disconnected_masks:
                out[semantic_id] = np.logical_or.reduce(disconnected_masks)
            else:
                out[semantic_id] = np.zeros_like(out[semantic_id], bool)

        prompt_points = []
        prompt_labels = []
        for semantic_id, svm_prediction in svm_predictions.items():
            prompt_point_id, _ = self.get_sam_prompts(svm_prediction.reshape((self.upsampled_h, self.upsampled_w)), out[semantic_id])
            prompt_points.append(prompt_point_id)

        prompt_points = np.concatenate(prompt_points, axis = 0).astype(np.float64)

        prompt_points[:,0] /= self.upsampled_w
        prompt_points[:, 1] /= self.upsampled_h

        mask_generator = SingleCropMaskGenerator(self.sam_predictor, points_per_side = None, point_grids=[prompt_points])
        mask_data = mask_generator.generate(img_square)

        masks = [mask["segmentation"] for mask in mask_data]

        combined_mask = np.zeros((self.upsampled_h, self.upsampled_w), dtype=np.int)

        assigned_masks, avg_preds = assign_masks(svm_predictions, masks)

        # current problem: Different numbers of train samples makes the svm predictions not comparable
        for semantic_id, pred in sorted(avg_preds.items(), key = lambda x: x[1]):
            if out[semantic_id].sum() == 0 or svm_predictions[semantic_id][assigned_masks[semantic_id]].sum() < 0.0:
                continue
            combined_mask[assigned_masks[semantic_id]] = semantic_id

        return combined_mask

    def combine_masks(self, mask_dict: dict):
        mask_shape = list(next(iter(mask_dict.values()))[0].shape)
        num_masks = len(mask_dict)

        logits_array = np.full(mask_shape + [num_masks], -np.inf)

        combined_mask = np.zeros(mask_shape, dtype=np.int)

        max_mask = np.zeros(mask_shape)
        for idx, (semantic_id, (mask, logits)) in enumerate(mask_dict.items()):
            index = idx + 1
            logits_array[mask, index-1] = logits[mask]

            max_mask[mask] = logits_array[mask,:].argmax(axis=-1) + 1
        
        for idx, semantic_id in enumerate(mask_dict.keys()):
            index = idx + 1
            combined_mask[max_mask == index] = semantic_id

        return combined_mask

    def load_dino_v2_model(self, dino_model_type: str) -> torch.nn.Module:
        model = torch.hub.load('facebookresearch/dinov2', dino_model_type)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    
    def preprocess_image_dino(self, image: np.ndarray) -> torch.Tensor:
        transform = T.Compose([
        T.Resize((self.patch_h * 14, self.patch_w * 14)),
        T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = transform(image).unsqueeze(0).cuda()
        return tensor
    
    def extract_features_dino(self, model, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = model.forward_features(input_tensor)['x_norm_patchtokens'].squeeze()
            # features = features.reshape(patch_h, patch_w, feat_dim)
            if self.upsampled_feature_vectors:
                features = features.reshape(self.patch_w, self.patch_h, -1).to(device=self.device)
                features = self.upsampler(features.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
                features = features.reshape(self.upsampled_h * self.upsampled_w, -1)

            # normalize features:

            features = features / torch.norm(features, dim=-1, keepdim=True)

        return features
    
    def compute_img_dino_embeddings(self, tensor_in: torch.Tensor, mask: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        self.upsampled_feature_vectors = False
        img_features = self.extract_features_dino(self.dino_model, tensor_in).to(device=self.device)
        if self.upsampled_feature_vectors:
            mask = cv2.GaussianBlur(mask.astype("float32"), (11, 11), 0) > 0.9
            mask_flat = mask.reshape(self.upsampled_h * self.upsampled_w)
        else:
            mask_sml = (cv2.resize(mask.squeeze().astype(float), (self.patch_w, self.patch_h), interpolation=cv2.INTER_AREA) > 0.8)
            mask_flat = mask_sml.reshape(self.patch_h * self.patch_w)

        self.upsampled_feature_vectors = True

        return img_features, mask_flat
    
    def initial_sam_prediction(self, 
                               x: int, y: int, 
                               img_square: np.ndarray, 
                               img: np.ndarray, 
                               embedding_sam: str, 
                               gt_mask: np.ndarray, 
                               prompts: dict = None) -> np.ndarray:
        if gt_mask is not None:
            mask = gt_mask
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 0.5
        elif prompts is not None:
            point_prompt = prompts['point_prompt']
            bbox_prompt = prompts['bbox']
            self.set_sam_img_embedding(img_square, embedding_sam)
            x = int(point_prompt[0] * self.upsampled_h / img.shape[1])
            y = int(point_prompt[1] * self.upsampled_w / img.shape[0])
            bbox_prompt = np.array([
                int(bbox_prompt[0] * self.upsampled_h / img.shape[1]),
                int(bbox_prompt[1] * self.upsampled_w / img.shape[0]),
                int(bbox_prompt[2] * self.upsampled_h / img.shape[1]),
                int(bbox_prompt[3] * self.upsampled_w / img.shape[0])])
            mask, iou_prediction, _ = self.sam_predictor.predict(
                point_coords=np.array([[x,y]]),
                point_labels=np.array([1]),
                box=bbox_prompt,
                multimask_output=False,
            )
            self.sam_predictor.reset_image()
        else: 
            with performance_measure("sam initial prediction"):
                x = int(x * self.upsampled_h / img.shape[1])
                y = int(y * self.upsampled_w / img.shape[0])

                self.set_sam_img_embedding(img_square, embedding_sam)

                mask, iou_prediction, _ = self.sam_predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    box=None,
                    multimask_output=False,
                )

                self.sam_predictor.reset_image()
        return mask
    
    def set_sam_img_embedding(self, image: np.ndarray, embedding: str = None) -> None:

        # get image embedding with precomputed embeddings, if available:
        if self.use_precomputed_sam_embedding and embedding is not None:
            if os.path.exists(embedding):
                input_image = self.sam_predictor.transform.apply_image(image)
                input_image_torch = torch.as_tensor(input_image, device=self.sam_predictor.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

                assert (
                    len(input_image_torch.shape) == 4
                    and input_image_torch.shape[1] == 3
                    and max(*input_image_torch.shape[2:]) == self.sam_predictor.model.image_encoder.img_size
                ), f"set_torch_image input must be BCHW with long side {self.sam_predictor.model.image_encoder.img_size}."
                self.sam_predictor.reset_image()

                self.sam_predictor.original_size = image.shape[:2]
                self.sam_predictor.input_size = tuple(input_image_torch.shape[-2:])
                self.sam_predictor.features = torch.load(embedding, map_location=self.sam_predictor.device)
                self.sam_predictor.is_image_set = True
            
            # no precomputed embeddings available:
            else:
                self.sam_predictor.set_image(image, image_format='BGR')

                if not os.path.exists(os.path.split(embedding)[0]):
                    os.makedirs(os.path.split(embedding)[0])

                features = self.sam_predictor.get_image_embedding()
                torch.save(features, embedding)
        else:
            self.sam_predictor.set_image(image, image_format='BGR')

    def create_svm_dataset(self, 
                           img_features: torch.Tensor, 
                           mask_flat: np.ndarray, 
                           dataset: str, 
                           task: str) -> Tuple[torch.utils.data.TensorDataset, torch.Tensor]:
        positive_features = img_features[mask_flat, :]
        all_negative_features_image = img_features[~mask_flat, :]
        negative_indices = torch.randperm(all_negative_features_image.shape[0])[:min(self.n_negative_features_same_image, all_negative_features_image.shape[0])]
        negative_features_image = all_negative_features_image[negative_indices]
        if dataset == "DAVIS_single_obj":
            negative_features = torch.cat([torch.cat(list(v.values())) for k,v in self.negative_sample_dict[dataset].items() if k != task]).to(device=self.device)
        else:
            negative_features = []
            for k,v in self.negative_sample_dict[dataset].items():
                scene_name = k.split("_")[0]+"_"+k.split("_")[1]
                object_name = k.split("_")[-1]
                if k.split("_")[2].isdigit():
                    scene_name = scene_name+"_"+k.split("_")[2]
                if scene_name not in task and object_name not in task:
                    negative_features.append(torch.cat(list(v.values())))
            negative_features = torch.cat(negative_features).to(device=self.device)

        all_negative_features = torch.cat((negative_features, negative_features_image))
        # all_negative_features = negative_features_image
        negative_labels = torch.zeros(all_negative_features.shape[0])
        positive_labels = torch.ones(positive_features.shape[0])

        features = torch.cat((positive_features, all_negative_features)).cuda()

        # normalize features:
        features = features / torch.norm(features, dim=-1, keepdim=True)

        labels = torch.cat((positive_labels, negative_labels)).cuda()

        class_counts = torch.bincount(labels.long())
        weights = 1.0 / class_counts[labels.long()]

        labels = labels * 2 - 1
        tensor_dataset = torch.utils.data.TensorDataset(features, labels)

        del features, labels

        return tensor_dataset, weights
    
    def train_svm(self, 
                  svm: nn.Linear, 
                  dataloader: torch.utils.data.dataloader.DataLoader) -> Tuple[nn.Linear, List[float]]:
        optimizer = optim.Adam(svm.parameters(), lr=0.1)
        loss_list = []
        for epoch in range(11):
            print("Epoch: ", epoch)
            loss_list_ep = []
            if epoch >= 6:
                optimizer.param_groups[0]['lr'] = 0.01
            for batch, target in tqdm(dataloader):
                optimizer.zero_grad()
                output = svm(batch).squeeze()
                weights = svm.weight.squeeze()
                hloss = self.loss(output, target, weights)
                loss_list_ep.append(hloss.item())
                hloss.backward()
                optimizer.step()
            print("Epoch_Loss", np.average(loss_list_ep))
            loss_list.append(np.average(loss_list_ep))

        return svm, loss_list
    
    def train_svm_conv(self, 
                       svm_conv: nn.Conv1d, 
                       dataloader: torch.utils.data.dataloader.DataLoader) -> Tuple[nn.Conv1d, List[float]]:
        optimizer = optim.Adam(svm_conv.parameters(), lr=0.1)
        loss_list = []
        for epoch in range(4):
            print("Epoch: ", epoch)
            loss_list_ep = []
            if epoch >= 8:
                optimizer.param_groups[0]['lr'] = 0.01
            for batch, target in tqdm(dataloader):
                optimizer.zero_grad()
                output = svm_conv(batch.unsqueeze(-1)).squeeze()
                weights = svm_conv.weight.squeeze()
                hloss = self.loss(output, target, weights)
                loss_list_ep.append(hloss.item())
                hloss.backward()
                optimizer.step()
            print("Epoch_Loss", np.average(loss_list_ep))
            loss_list.append(np.average(loss_list_ep))
        
        return svm_conv, loss_list
    
    def show_mask(self, 
                  mask: np.ndarray, 
                  color: np.ndarray = None) -> np.ndarray:
        if color is not None:
            color = color
        else:
            color = np.array([0, 0, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image.astype("uint8")
    
    def get_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        mask = mask.squeeze().astype(np.uint8)
        if np.sum(mask) == 0:
            return None
        
        row_indices, col_indices = np.where(mask)

        # Calculate the min and max row and column indices
        row_min, row_max = np.min(row_indices), np.max(row_indices)
        col_min, col_max = np.min(col_indices), np.max(col_indices)

        # Return the bounding box coordinates as a tuple
        return (col_min, row_min, col_max, row_max)

    def visualize_svm_distance(self, 
                               svm_predictions: np.ndarray, 
                               img: np.ndarray, 
                               semantic_id: int) -> None:

        if self.upsampled_feature_vectors:
            svm_dist = svm_predictions.reshape((self.upsampled_h, self.upsampled_w))
        else:
            svm_dist = svm_predictions.reshape((self.patch_h, self.patch_w))

        vis_margin = (self.margins[semantic_id]/2 - svm_dist.min()) / (svm_dist.max() - svm_dist.min())

        # resize to image size:
        svm_dist = cv2.resize(svm_dist, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        svm_dist = svm_dist.astype("float32")
        svm_dist = (svm_dist - svm_dist.min()) / (svm_dist.max() - svm_dist.min())

        margin = (np.logical_and(svm_dist <= vis_margin*1.2, svm_dist >= vis_margin))
        margin2 = (np.logical_and(svm_dist <= vis_margin*1.6, svm_dist >= vis_margin*1.5))

        svm_dist = cv2.cvtColor(svm_dist, cv2.COLOR_GRAY2BGR)
        svm_dist[:, 1] = 0
        svm_dist[:, 2] = 0
        svm_dist[margin, 2] = 1.0
        svm_dist[margin2, 1] = 1.0
        
        cv2.imshow("SVM Dist", cv2.addWeighted(img, 0.2, (svm_dist*255).astype("uint8"), 0.8, 0.0))
    
    def sam_refinement(self, 
                       img: np.ndarray, 
                       mask: np.ndarray, 
                       svm_predictions: np.ndarray) -> np.ndarray:

        prompt_points, prompt_labels = self.get_sam_prompts(svm_predictions, mask)
        #could also sample positive points from the mask and negative from outside the mask
        bbox = np.array(self.get_bbox_from_mask(mask))
        mask = cv2.resize(mask.astype("float32"), (256, 256)) > 0.5

        mask_refined, iou_prediction, logits = self.sam_predictor.predict(
                    # mask_input=np.expand_dims(mask * 20 - 10, axis=0),
                    point_coords=prompt_points,
                    point_labels=prompt_labels.T,
                    multimask_output=False,
                )

        return mask_refined.squeeze(0)

    def one_hot_encode_binary_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        num_labels, labeled_mask, stats, centroid = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        total_area = np.sum((mask > 0))
        masks_ = [(labeled_mask == i) for i in np.unique(labeled_mask) if i != 0 and np.sum((labeled_mask == i)) > total_area * 0.2]

        return masks_
    
    def get_sam_prompts(self, 
                        svm_predictions: np.ndarray, 
                        mask: np.ndarray, 
                        neighborhood_size: int = 5, 
                        n_maxima: int = 8, 
                        n_negative_points: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        neighborhood_size = neighborhood_size
        n_maxima = n_maxima
        n_negative_points = n_negative_points

        data_max = maximum_filter(svm_predictions, size=neighborhood_size)
        maxima = (svm_predictions == data_max)
        
        structure = iterate_structure(generate_binary_structure(2, 2), 1)
        maxima = maxima ^ binary_erosion(maxima, structure, border_value=1)

        coordinates = np.where(maxima)

        maxima_values = svm_predictions[coordinates]

        # Get the indices of the top n maxima
        top_n_indices = np.argpartition(maxima_values, -n_maxima)[-n_maxima:]

        # Get the coordinates of the top n maxima
        top_n_coordinates = coordinates[0][top_n_indices], coordinates[1][top_n_indices]

        positive_points = np.array(top_n_coordinates).T
        positive_points = positive_points[:,(-1,0)]
        positive_points = [p for p in positive_points if mask[p[1], p[0]]] + [np.array([svm_predictions.argmax()%512, svm_predictions.argmax()//512])]
        positive_points = np.array(positive_points)
        positive_labels = np.ones(len(positive_points))

        negative_sample_space = svm_predictions < 0.0
        negative_points = np.array(np.where(negative_sample_space)).T
        #random_subset:
        negative_points = negative_points[np.random.choice(negative_points.shape[0], n_negative_points, replace=False)]
        negative_points = negative_points[:,(-1,0)]
        negative_labels = np.zeros(len(negative_points))
        return np.concatenate([positive_points, negative_points], axis=0), np.concatenate([positive_labels, negative_labels], axis=0)
