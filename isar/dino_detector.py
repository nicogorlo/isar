import numpy as np
import os
from PIL import Image
import cv2
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler,Sampler
from torchmetrics import HingeLoss


from sklearn.decomposition import PCA

from params import OUTDIR
from util.isar_utils import performance_measure, semantic_obs_to_img, generate_pastel_color
from sam_mask_generator import SingleCropMaskGenerator
from segment_anything import SamPredictor, sam_model_registry


class DinoDetector():
    def __init__(self, device, sam_model_type, dino_model_type = 'dinov2_vitl14', use_precomputed_sam_embeddings = False, outdir = OUTDIR, n_per_side = 16) -> None:
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
        self.single_crop_mask_generator = SingleCropMaskGenerator(self.sam_predictor, points_per_side = n_per_side)

        # DINO model:
        self.dino_model = self.load_dino_v2_model(dino_model_type).to(device='cuda')

        self.patch_h = 40
        self.patch_w = 40
        self.upsampled_h = 512
        self.upsampled_w = 512

        self.upsampled_feature_vectors = True

        self.upsampler = nn.Upsample(size=(self.upsampled_h, self.upsampled_w), mode='bilinear', align_corners=False)

        # classifier:
        self.svm = LinearClassifier(self.dino_model.num_features, 1).to(device='cuda')
        self.loss = nn.HingeEmbeddingLoss(reduction='mean').to(device='cuda')
        self.optimizer = optim.SGD(self.svm.parameters(), lr=0.1, momentum=0.9)

        self.svm_conv = nn.Conv2d(self.dino_model.num_features, 1, kernel_size=1, stride=1, padding=0, device="cuda")

        self.n_negative_features_same_image = 200

        if self.upsampled_feature_vectors:
            self.n_negative_features_same_image *= 16

        with open(f"feature_dict_dinov2/dino_features_{dino_model_type}.pkl", "rb") as f:
            self.negative_sample_dict = pickle.load(f)

        self.margin = 0.0
        
        # self.sampler = BatchSampler()
        # self.dataloader = DataLoader(batch_size=32, shuffle=True, num_workers=2)

        #visualizations:
        self.semantic_palette = np.array([generate_pastel_color() for _ in range(256)],dtype=np.uint8)


    def on_click(self, x: int, y: int, img: np.ndarray, embedding_sam: str, dataset = 'DAVIS_single_obj',task: str = 'bear', gt_mask = None):
        self.start_reid = True
        torch.set_grad_enabled(True)
        # 1. predict mask with sam_predictor
        if gt_mask is not None:
            mask = gt_mask
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 0.5
            sam_img = cv2.resize(img, (self.upsampled_h, self.upsampled_w), interpolation=cv2.INTER_NEAREST)
        else: 
            with performance_measure("sam initial prediction"):
                x = int(x * self.upsampled_h / img.shape[1])
                y = int(y * self.upsampled_w / img.shape[0])
                sam_img = cv2.resize(img, (self.upsampled_h, self.upsampled_w))

                self.set_sam_img_embedding(sam_img, embedding_sam)

                mask, iou_prediction, _ = self.sam_predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    box=None,
                    multimask_output=False,
                )

        with performance_measure("preprocess dino image"):
            # 2. preprocess image dino
            tensor = self.preprocess_image_dino(img)


        with performance_measure("masked dino embeddings"):
            # 3. compute masked dino embeddings
            img_features = self.extract_features_dino(self.dino_model, tensor).to(device=self.device)
            if self.upsampled_feature_vectors:
                mask_flat = mask.reshape(self.upsampled_h * self.upsampled_w) > 0.5
            else:
                mask_sml = (cv2.resize(mask.squeeze().astype(float), (self.patch_w, self.patch_h)) > 0.5)
                mask_flat = mask_sml.reshape(self.patch_h * self.patch_w)

        with performance_measure("create dataset"):
            # 4. get negative features from dataset
            positive_features = img_features[mask_flat, :]
            all_negative_features_image = img_features[~mask_flat, :] ### TODO: try to use all negative features from the same image
            negative_indices = torch.randperm(all_negative_features_image.shape[0])[:min(self.n_negative_features_same_image, all_negative_features_image.shape[0])]
            negative_features_image = all_negative_features_image[negative_indices]
            negative_features = torch.cat([torch.cat(list(v.values())) for k,v in self.negative_sample_dict[dataset].items() if k != task]).to(device=self.device)

            all_negative_features = torch.cat((negative_features, negative_features_image))
            negative_labels = torch.zeros(all_negative_features.shape[0])
            positive_labels = torch.ones(positive_features.shape[0])


            features = torch.cat((positive_features, all_negative_features)).cuda()

            # normalize features:
            features = features / torch.norm(features, dim=1, keepdim=True)

            #min-max normalization:
            # features = ((features - features.min(dim=1).values.unsqueeze(-1)) / 
            #         (features.max(dim=1).values.unsqueeze(-1) - features.min(dim=1).values.unsqueeze(-1)))

            labels = torch.cat((positive_labels, negative_labels)).cuda()

            class_counts = torch.bincount(labels.long())
            weights = 1.0 / class_counts[labels.long()]

            labels = labels * 2 - 1
            tensor_dataset = torch.utils.data.TensorDataset(features, labels)
            num_samples = len(tensor_dataset)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)

            dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=2048, sampler = sampler, num_workers=0)

        with performance_measure("train svm"):
            # 5. fit SVM to embeddings (+) and [random stored embeddings + embeddings outside mask] (-)
            #   * Single Conv1d layer with 1x1 kernel
            #   * 1x1 kernel is a linear classifier
            #   * hinge loss
            #   * BatchSGD optimizer, make sure classes in batches are balanced
            loss_list = []
            for epoch in range(20):
                print("Epoch: ", epoch)
                # epoch_loss = self.train_svm(dataloader)
                loss_list_ep = []
                if epoch >= 8:
                    self.optimizer.param_groups[0]['lr'] = 0.01
                for batch, target in tqdm(dataloader):
                    self.optimizer.zero_grad()
                    # output = self.svm_conv(batch.transpose(0,1).unsqueeze(-1)).squeeze()
                    output = self.svm(batch).squeeze()
                    hloss = torch.mean(torch.clamp(1 - target * output, min=0))
                    loss_list_ep.append(hloss.item())
                    hloss.backward()
                    self.optimizer.step()
                print("Epoch_Loss", np.average(loss_list_ep))

        self.margin = 2.0 / np.linalg.norm(self.svm.linear.weight.detach().cpu().numpy())
        
        # TEST inference:
        # with performance_measure("inference:"):
        #     tensor = self.preprocess_image_dino(img).cuda()
        #     img_features = self.extract_features_dino(self.dino_model, tensor)
        #     img_features = img_features / torch.norm(img_features, dim=1, keepdim=True)
        #     svm_predictions = self.svm(img_features).detach().cpu().numpy()
        #     out = (svm_predictions > self.margin/2).reshape((self.patch_h, self.patch_w))
        #     out = cv2.resize(out.astype(float), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype("float64")

        bgr_mask = self.show_mask(cv2.resize(mask.squeeze().astype(float), (img.shape[1], img.shape[0]))).astype("uint8")

        dst = cv2.addWeighted(img.astype("uint8"), 0.7, bgr_mask.astype("uint8"), 0.3, 0).astype("uint8")
        cv2.imshow("seg", dst)
        cv2.waitKey(1000)
        # not functional yet:

        # with performance_measure("predict conv2d"):
        #     self.svm_conv.weight.data = self.svm.linear.weight.view(1, self.dino_model.num_features, 1, 1)
        #     self.svm_conv.bias.data = self.svm.linear.bias
        #     # 6. transform embeddings with SVM to get (40x40) predictions
        #     img_features = img_features.reshape(1, self.dino_model.num_features, self.patch_h, self.patch_w).cpu()
        #     svm_predictions = self.svm_conv(img_features).detach().squeeze().cpu().numpy()
        #     # svm_predictions = torch.sigmoid(svm_predictions).detach().cpu().numpy()
        #     out = svm_predictions < -12000
        #     cv2.imshow("svm_predictions", out.astype("float64"))
        #     cv2.waitKey(0)

        bbox = self.get_bbox_from_mask(mask)
        if bbox is not None:
            cutout = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        else:
            cutout = None

        torch.set_grad_enabled(False)
        return cutout, bgr_mask, img, 1.0, bbox

    def detect(self, img: np.ndarray, image_name: str, embedding: str):

        scores = None
        boxes = None
        show_mask = None

        if self.start_reid:
            tensor = self.preprocess_image_dino(img).cuda()
            img_features = self.extract_features_dino(self.dino_model, tensor).cuda()
            svm_predictions = self.svm(img_features).detach().cpu().numpy()
            # svm_predictions = self.svm_conv(img_features.transpose(0,1).unsqueeze(-1)).detach().cpu().numpy()
            if self.upsampled_feature_vectors:
                out = (svm_predictions > self.margin/2).reshape((self.upsampled_h, self.upsampled_w))
            else:
                out = (svm_predictions > self.margin/2).reshape((self.patch_h, self.patch_w))
            out = cv2.resize(out.astype(float), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype("float64")

            boxes = [self.get_bbox_from_mask(out)]
            scores = [1.0]
            show_mask = self.show_mask(out).astype("uint8")

            dst = cv2.addWeighted(img.astype('uint8'), 0.7, show_mask.astype('uint8'), 0.3, 0).astype('uint8')

            if self.show_images:
                cv2.imshow("seg", dst)

            cv2.imwrite(os.path.join(self.outdir, image_name), dst)

        return scores, boxes, show_mask
    

    def load_dino_v2_model(self, dino_model_type):
        model = torch.hub.load('facebookresearch/dinov2', dino_model_type)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    
    def preprocess_image_dino(self, image: np.ndarray):
        transform = T.Compose([
        T.Resize((self.patch_h * 14, self.patch_w * 14)),
        T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = transform(image).unsqueeze(0).cuda()
        return tensor
    
    def extract_features_dino(self, model, input_tensor):
        with torch.no_grad():
            features_dict = model.forward_features(input_tensor)
            features = features_dict['x_norm_patchtokens'].squeeze()
            # features = features.reshape(patch_h, patch_w, feat_dim)
            if self.upsampled_feature_vectors:
                features = features.reshape(self.patch_w, self.patch_h, -1).to(device=self.device)
                features = self.upsampler(features.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
                features = features.reshape(self.upsampled_h * self.upsampled_w, -1)

            # normalize features:
            features = features / torch.norm(features, dim=1, keepdim=True)

        return features
    
    def set_sam_img_embedding(self, image: np.ndarray, embedding = None):

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
    
    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image
    
    def get_bbox_from_mask(self, mask: np.ndarray):
        mask = mask.squeeze().astype(np.uint8)
        if np.sum(mask) == 0:
            return None
        
        row_indices, col_indices = np.where(mask)

        # Calculate the min and max row and column indices
        row_min, row_max = np.min(row_indices), np.max(row_indices)
        col_min, col_max = np.min(col_indices), np.max(col_indices)

        # Return the bounding box coordinates as a tuple
        return (col_min, row_min, col_max, row_max)
        
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def main():
    detector = DinoDetector("cpu", "vit_h", 'dinov2_vitl14', use_precomputed_sam_embeddings=True)
    task = 'bear'
    dataset = 'DAVIS_single_obj'
    taskdir = os.path.join('/home/nico/semesterproject/data/DAVIS_single_object_tracking')
    imgdir = os.path.join(taskdir, task, 'rgb/')
    embdir = os.path.join(taskdir, task, 'embeddings/')
    evaldir = os.path.join(taskdir, task, 'semantics/')
    with open(os.path.join(taskdir, 'prompt_dict.json'), 'r') as f:
            prompt_dict = json.load(f)
    
    image_names = sorted(os.listdir(imgdir))
    img0 = cv2.imread(os.path.join(imgdir, image_names[0]))
    emb0 = os.path.join(embdir, image_names[0].replace(".jpg", ".pt"))

    prompt = prompt_dict[task]
    cutout, seg, freeze, selected_prob, selected_box = detector.on_click(
        x = prompt['x'], y = prompt['y'], img = img0, embedding_sam=emb0, 
        dataset=dataset, task=task)
    
    img = cv2.imread("/home/nico/semesterproject/data/DAVIS_single_object_tracking/rhino/rgb/0000020.jpg")
    detector.detect(img, "00000020.jpg", "/home/nico/semesterproject/data/DAVIS_single_object_tracking/rhino/embeddings/0000020.pt")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()