import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util import box_ops
from models import build_model
import time


from types import SimpleNamespace

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


    """
    Args: 
    * im: PIL image
    Returns:
    * bboxes_scaled: list of bounding boxes in format [x1, y1, x2, y2]
    * scores: list of scores
    * labels: list of labels
    * keep: list of booleans indicating whether the box is kept
    """
    def __call__(self, im):
        im = Image.fromarray(im.astype('uint8'), 'RGB')
        img = transform(im).unsqueeze(0)
        img=img.cuda()
        # propagate through the model
        outputs = self.model(img)

        # label prediction (in our case unused)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = torch.nn.functional.softmax(out_logits, -1)[0, :, :-1]
        keep = prob.max(-1).values > self.keep_logits
        

        # select 100 best boxes based on objectness score
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
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

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def rescale_bboxes(self, out_bbox: torch.tensor, size: tuple) -> torch.tensor:
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b




# def main(args):
#     utils.init_distributed_mode(args)
#     print("git:\n  {}\n".format(utils.get_sha()))

#     if args.frozen_weights is not None:
#         assert args.masks, "Frozen training is meant for segmentation only"
#     print(args)

#     device = torch.device(args.device)

#     # # fix the seed for reproducibility
#     # seed = args.seed + utils.get_rank()
#     # torch.manual_seed(seed)
#     # np.random.seed(seed)
#     # random.seed(seed)

#     def box_cxcywh_to_xyxy(x: torch.tensor) -> torch.tensor:
#         x_c, y_c, w, h = x.unbind(1)
#         b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#             (x_c + 0.5 * w), (y_c + 0.5 * h)]
#         return torch.stack(b, dim=1)

#     model, criterion, postprocessors = build_model(args)
#     model.to(device)
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     print(checkpoint.keys())
#     model.load_state_dict(checkpoint['model'], strict=False)
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()

#     t0 = time.time()
#     im = Image.open(args.img_path)
#     # mean-std normalize the input image (batch-size: 1)
#     img = transform(im).unsqueeze(0)

#     img=img.cuda()
#     # propagate through the model
#     outputs = model(img)

#     out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

#     prob = out_logits.sigmoid()
#     topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
#     scores = topk_values
#     topk_boxes = topk_indexes // out_logits.shape[2]
#     labels = topk_indexes % out_logits.shape[2]
#     boxes = box_cxcywh_to_xyxy(out_bbox)
#     boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

#     keep = scores[0] > 0.35

#     print("scores: ", scores)
#     print("scores[0]: ", scores[0])
#     print("scores[0], keep: ", scores[0, keep])
#     boxes = boxes[0, keep]
#     labels = labels[0, keep]

#     # and from relative [0, 1] to absolute [0, height] coordinates
#     im_h,im_w = im.size
#     #print('im_h,im_w',im_h,im_w)
#     target_sizes =torch.tensor([[im_w,im_h]])
#     target_sizes =target_sizes.cuda()
#     img_h, img_w = target_sizes.unbind(1)
#     scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#     boxes = boxes * scale_fct[:, None, :]
#     print(time.time()-t0)
#     #plot_results
#     source_img = Image.open(args.img_path).convert("RGBA")

#     draw = ImageDraw.Draw(source_img)
    
#     print("Boxes",boxes,boxes.tolist())
#     for xmin, ymin, xmax, ymax in boxes[0].tolist():
#         draw.rectangle(((xmin, ymin), (xmax, ymax)), outline ="red")

#     source_img.save('test.png', "png")
#     results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
#     #print("Outputs",results)


# if __name__ == '__main__':
#     args = argparse.Namespace()
#     args_dict = vars(args)
#     with open('args.json', 'r') as f:
#         args_dict_new = json.load(f)

#     args_dict.update(args_dict_new)
    
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)