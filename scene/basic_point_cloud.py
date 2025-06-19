import numpy as np


class BasicPointCloud:
    """
    Point cloud.

    Parameters
    ----------
    positions : np.ndarray
        Positions for points (3D coordinates).
    colors : np.ndarray
        Colors for points (RGB).
    normals : np.ndarray
        Normals for points.
    """
    def __init__(self,
                 positions: np.ndarray,
                 colors: np.ndarray,
                 normals: np.ndarray | None = None):
        super(BasicPointCloud, self).__init__()
        assert (positions.shape == colors.shape)
        assert (normals is None) or (normals.shape == positions.shape)
        assert (positions.shape[1] == 3)
        assert (colors.min().item() >= 0.0)
        assert (colors.max().item() <= 1.0)

        self.positions = positions
        self.colors = colors
        self.normals = normals
