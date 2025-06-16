import numpy as np


class BasicPointCloud:
    def __init__(self,
                 points: np.ndarray,
                 colors: np.ndarray,
                 normals: np.ndarray):
        super(BasicPointCloud, self).__init__()
        self.points = points
        self.colors = colors
        self.normals = normals
