import numpy as np
import cv2

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
        self.img_embedding_path = None

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.predictor = SamPredictor(self.sam)

    def __call__(self, image: np.ndarray, bbox: np.array) -> np.ndarray:

        with performance_measure("segmentation"):
            self.predictor.set_image(image)

            mask, _, _ = self.predictor.predict(
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


    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:

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
    
    def normPRED(self,d: torch.tensor) -> torch.tensor:
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn