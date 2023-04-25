import numpy as np
import cv2
import os

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import PIL
from PIL import Image

from util.isar_utils import performance_measure

from segment_anything import sam_model_registry, SamPredictor


class SegmentorSAM():
    def __init__(self, device, model_type) -> None:
        device = device  ## RIGHT now running on CPU, as Laptop GPU memory not sufficient
        model_type = model_type
        sam_checkpoint = os.path.join('modelzoo', [i for i in os.listdir('modelzoo') if model_type in i][0])

        self.use_precomputed_embedding = False

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
        self.predictor = SamPredictor(self.sam)

    def __call__(self, image: np.ndarray, bbox: np.array, embedding = None) -> np.ndarray:
        """
        Args:
            image (np.ndarray): image to segment
            bbox (np.ndarray): bounding box of object to segment
            embedding (str): path to precomputed embedding
        Returns:
            mask_img (np.ndarray): three channel image of mask

        
        """

        with performance_measure("segmentation"):

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

                    if not os.path.exists(embedding):
                        features = self.predictor.get_image_embedding()
                        torch.save(features, embedding)

            else:
                self.predictor.set_image(image, image_format='BGR')

            # mask (np.ndarray): binary mask of shape (1, H, W)
            # quality (float): quality of the mask [0, 1]
            # mask_lowres: np.ndarray, binary mask of the object in a resized image of shape (1, 256, 256)
            mask, mask_qualities, mask_lowres = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox,
                multimask_output=False,
            )
       
        mask_img = self.show_mask(mask, random_color=False)

        return mask_img

    def show_mask(self, mask: np.ndarray, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([0, 0, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        return mask_image



from external.u2net.model import U2NET # full size version 173.6 MB
from external.u2net.data_loader import RescaleT
from external.u2net.data_loader import ToTensorLab
from external.u2net.data_loader import ToTensor
from external.u2net.data_loader import SalObjDataset
from external.u2net.data_loader import DataLoader
from torch.autograd import Variable
from torchvision import transforms



class SegmentorU2():
    def __init__(self) -> None:

        model_dir = 'modelzoo/u2net.pth'

        self.net = U2NET(3, 1)
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.net.eval()

        self.transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])


    def __call__(self, image: np.ndarray, bbox: np.ndarray, emb: torch.Tensor) -> np.ndarray:

        cutout = self.get_cutout(image, bbox)

        img = torch.from_numpy(cutout).float()

        sample = self.transform({'imidx': np.array([0]), 'image': img, 'label': img})

        img = sample['image'].unsqueeze(0)

        if torch.cuda.is_available():
            img = Variable(img.cuda())
        else:
            img = Variable(img)

        d1,d2,d3,d4,d5,d6,d7= self.net(img.float())
        
        pred = d1[:,0,:,:]
        pred = self.normPRED(pred)
        predict = pred
        # predict = predict.squeeze()
        predict_np = pred.cpu().data.numpy().squeeze()

        im = Image.fromarray(predict_np*255).convert('RGB')

        im = np.array(im.resize((cutout.shape[1],cutout.shape[0]),resample=Image.BILINEAR))

        return im
    
    def get_cutout(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    def normPRED(self,d: torch.Tensor) -> torch.Tensor:
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn