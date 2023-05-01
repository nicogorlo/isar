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
        self.dino_model = self.load_dino_v2_model(dino_model_type)

        self.patch_h = 40
        self.patch_w = 40

        # classifier:
        self.svm = LinearClassifier(self.dino_model.num_features, 1)
        self.loss = nn.HingeEmbeddingLoss(reduction='mean')
        self.optimizer = optim.Adam(self.svm.parameters(), lr=0.01)#, momentum=0.9)

        self.svm_conv = nn.Conv2d(self.dino_model.num_features, 1, kernel_size=1, stride=1, padding=0, device=self.device)

        self.n_negative_features_same_image = 200
        with open(f"../../data/dino_features_{dino_model_type}.pkl", "rb") as f:
            self.negative_sample_dict = pickle.load(f)
        

        
        # self.sampler = BatchSampler()
        # self.dataloader = DataLoader(batch_size=32, shuffle=True, num_workers=2)

        #visualizations:
        self.semantic_palette = np.array([generate_pastel_color() for _ in range(256)],dtype=np.uint8)


    def on_click(self, x: int, y: int, img: np.ndarray, embedding_sam: str, dataset = 'DAVIS_single_obj',task: str = 'bear'):

        # 1. predict mask with sam_predictor
        with performance_measure("sam initial prediction"):
            x = int(x * 512 / img.shape[1])
            y = int(y * 512 / img.shape[0])
            sam_img = cv2.resize(img, (512, 512))

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


        with performance_measure("masekd_dino_embeddings"):
            # 3. compute masked dino embeddings
            img_features = self.extract_features_dino(self.dino_model, tensor)
            mask_sml = (cv2.resize(mask.squeeze().astype(float), (self.patch_w, self.patch_h)) > 0.5)
            mask_sml = mask_sml.reshape(self.patch_h * self.patch_w)

            positive_features = img_features[mask_sml, :].cpu()

        with performance_measure("create dataset"):
            # 4. get negative features from dataset
            all_negative_features_image = img_features[~mask_sml, :]
            negative_indices = torch.randperm(all_negative_features_image.shape[0])[:self.n_negative_features_same_image]
            negative_features_image = all_negative_features_image[negative_indices]
            negative_features = torch.cat([torch.cat(list(v.values())) for k,v in self.negative_sample_dict[dataset].items() if k != task])

            all_negative_features = torch.cat((negative_features, negative_features_image)).cpu()
            negative_labels = torch.zeros(all_negative_features.shape[0]).cpu()
            positive_labels = torch.ones(positive_features.shape[0]).cpu()


            features = torch.cat((positive_features, all_negative_features))
            labels = torch.cat((positive_labels, negative_labels))
            tensor_dataset = torch.utils.data.TensorDataset(features, labels)
        
            num_samples = len(tensor_dataset)
            class_counts = torch.bincount(labels.long())
            weights = 1.0 / class_counts[labels.long()]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)

            dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1024, sampler = sampler, num_workers=8)

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
                for batch, target in tqdm(dataloader):
                    self.optimizer.zero_grad()
                    output = self.svm(batch)
                    hloss = self.loss(output, target)
                    loss_list_ep.append(hloss.item())
                    hloss.backward()
                    self.optimizer.step()
                print("Epoch_Loss", np.average(loss_list_ep))
        

        #option1: 
        # TODO: need better SVM prediction rule, maybe do normalization and centering before
        with performance_measure("predict pixelwise"):
            svm_predictions = self.svm(img_features.cpu()).detach().cpu().numpy()
            out = (svm_predictions < -12000).reshape((self.patch_h, self.patch_w))

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

        #return cutout, show_mask, freeze, selected_prob, selected_box
        pass

    def detect(self, img: np.ndarray, image_name: str, embedding: str):
        
        # 1. preprocess image 
        tensor = self.preprocess_image_dino(img)
        # 2. compute masked dino embeddings
        img_features = self.extract_features_dino(self.dino_model, tensor)

        svm_predictions = self.svm(img_features.cpu()).detach().cpu().numpy()
        out = (svm_predictions < -12000).reshape((self.patch_h, self.patch_w))
        cv2.imshow("svm_predictions", out.astype("float64"))
        cv2.waitKey(1)

        # return scores, boxes, show_mask
        pass
    

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
            features = features_dict['x_norm_patchtokens']
            # features = features.reshape(patch_h, patch_w, feat_dim)
        return features.squeeze()
    
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
    
    def train_svm(self, dataloader):
        loss_list_ep = []

        for batch, target in tqdm(dataloader):
            output = self.svm(batch)
            hloss = self.loss(output.squeeze(), target)
            loss_list_ep.append(hloss.item())
            hloss.backward()
            self.optimizer.step()

        return loss_list_ep
        
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
    
    print("Cutout: ", cutout)
    

if __name__ == "__main__":
    main()