import numpy as np


class BasicPointCloud:
    def __init__(self,
                 points: np.ndarray,
                 colors: np.ndarray,
                 normals: np.ndarray | None = None):
        super(BasicPointCloud, self).__init__()
        assert (points.shape == colors.shape)
        assert (normals is None) or (normals.shape == points.shape)
        assert (points.shape[1] == 3)
        assert (colors.min().item() >= 0.0)
        assert (colors.max().item() <= 1.0)

        self.points = points
        self.colors = colors
        self.normals = normals
