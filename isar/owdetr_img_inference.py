import argparse
import json
from PIL import Image
import torchvision.transforms as T

import numpy as np
import torch
import util.misc as utils
from models import build_model

from params import NUM_BOXES

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class OW_DETR():
    def __init__(self, keep_logits=0.7, keep_objectness=0.0):
        self.keep_logits = keep_logits
        self.keep_objectness = keep_objectness

        self.args = argparse.Namespace()
        args_dict = vars(self.args)
        with open('args.json', 'r') as f:
            args_dict_new = json.load(f)
        args_dict.update(args_dict_new)

        # not used
        utils.init_distributed_mode(self.args)
        print("git:\n  {}\n".format(utils.get_sha()))

        self.device = torch.device(self.args.device)

        self.model, self.criterion, self.postprocessors = build_model(self.args)
        self.model.to(self.device)
        checkpoint = torch.load(self.args.resume, map_location='cpu')
        print(checkpoint.keys())
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()



    def __call__(self, im: np.ndarray):
        im = Image.fromarray(im.astype('uint8'), 'RGB')
        img = transform(im).unsqueeze(0)
        img=img.cuda()

        # propagate through the model
        outputs = self.model(img)

        # label prediction (in our case unused so far)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = torch.nn.functional.softmax(out_logits, -1)[0, :, :-1]
        keep = prob.max(-1).values > self.keep_logits
        
        # select 100 best boxes based on objectness score
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), NUM_BOXES, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        keep = scores[0] > self.keep_objectness
        # boxes = boxes[0, keep]
        # labels = labels[0, keep]

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h,im_w = im.size
        #print('im_h,im_w',im_h,im_w)
        target_sizes =torch.tensor([[im_w,im_h]])
        target_sizes =target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        return scores, boxes, labels, keep

    def box_cxcywh_to_xyxy(self, x: torch.tensor) -> torch.tensor:
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def rescale_bboxes(self, out_bbox: torch.tensor, size: tuple) -> torch.tensor:
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
