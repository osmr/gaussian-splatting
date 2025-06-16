import numpy as np


class ColmapPointCloud:
    def __init__(self,
                 points: np.ndarray,
                 colors: np.ndarray,
                 errors: np.ndarray):
        super(ColmapPointCloud, self).__init__()
        assert (points.shape == colors.shape)
        assert (errors.shape[0] == points.shape[0])
        assert (points.shape[1] == 3)
        assert (errors.shape[1] == 1)
        assert (colors.min().item() >= 0)
        assert (colors.max().item() <= 255)

        self.points = points
        self.colors = colors
        self.errors = errors
