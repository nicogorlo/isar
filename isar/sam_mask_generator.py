import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from typing import List, Tuple

from segment_anything.utils.amg import MaskData, uncrop_boxes_xyxy, uncrop_points, batch_iterator
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore


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

