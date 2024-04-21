import numpy as np
import torch

class ImageTensor:

    def __init__(self, tensor, th=1):
        """
        Load an image tensor and a threshold for blank space detection.
        Threshold is defaulted to 1 because of Normalization in the
        albumentations pipeline.
        """
        self.arr = tensor.permute(1,2,0).numpy()
        self.th = th


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
            lines.append(line)
        return lines