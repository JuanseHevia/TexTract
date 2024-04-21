from typing import Tuple
import cv2
import numpy as np
import torch

def convert_to_candidate_size(arr, candidates):
    """
    Given an array representing an image of (H, W) dimensions,
    match the closest candidate size from the list of candidate sizes.
    The list is ordered in ascending order, and the closest candidate size
    is the first tuple that is above both height and width values
    """
    H, W, _ = arr.shape
    new_shape = arr.shape
    for candidate in candidates:
        if candidate[1] >= H and candidate[0] >= W:
            new_w = candidate[0]
            new_h = candidate[1]
            break
    
    # pad the image 
    new_arr = np.ones((new_h, new_w, 1)) * -1
    new_arr[:H, :W, :] = arr
    return new_arr
    

class ImageTensor:

    def __init__(self, tensor, candidate_sizes, th=1):
        """
        Load an image tensor and a threshold for blank space detection.
        Threshold is defaulted to 1 because of Normalization in the
        albumentations pipeline.
        """
        self.arr = tensor.permute(1,2,0).numpy()
        self.th = th
        self.candidate_sizes = candidate_sizes


    def _find_blank_space(self):
        """
        Spot longitudinal blank spaces in the image
        """

        x = self.arr.sum(2)
        yblanks = np.argwhere(x.min(1) > self.th*3).flatten()
        self.yblanks = yblanks

    def _find_filled_spaces(self):
        """
        Spot longitudinal filled spaces in the image
        """
        x = self.arr.sum(2)
        yfilled = np.argwhere(x.min(1) > self.th*3).flatten()
        self.yfilled = yfilled

    def _aggregate_vertical_indices(self):
        """
        Given a list of vertical indices, aggregate them into
        single lines where they are contiguous
        """
        yblanks = np.sort(self.yblanks)
        yblanks = np.split(yblanks, np.where(np.diff(yblanks) > 1)[0] + 1)
        return [np.mean(yblank).astype(int) for yblank in yblanks]

    def split_img_into_lines(self):
        """
        Given a multiline image, comupute blank line separators
        and return the resulting line formulas in a list
        """
        self._find_blank_space()
        agg_blanks = self._aggregate_vertical_indices()
        lines = []
        for i, yblank in enumerate(agg_blanks):
            if i == 0:
                continue
            line = self.arr[agg_blanks[i-1]:yblank]

            # resize to meet model dimension constraints
            line = convert_to_candidate_size(line, candidates=self.candidate_sizes)
            
            # convert to Tensor
            line = torch.tensor(line, dtype=torch.float32).permute(2,0,1)
            lines.append(line)
        return lines