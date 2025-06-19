import numpy as np


class ColmapPointCloud:
    """
    Colmap specific point cloud.

    Parameters
    ----------
    positions : np.ndarray
        Positions for points (3D coordinates).
    colors : np.ndarray
        Colors for points (RGB).
    errors : np.ndarray
        Errors for points.
    """
    def __init__(self,
                 positions: np.ndarray,
                 colors: np.ndarray,
                 errors: np.ndarray):
        super(ColmapPointCloud, self).__init__()
        assert (positions.shape == colors.shape)
        assert (errors.shape[0] == positions.shape[0])
        assert (positions.shape[1] == 3)
        assert (errors.shape[1] == 1)
        assert (colors.min().item() >= 0)
        assert (colors.max().item() <= 255)

        self.positions = positions
        self.colors = colors
        self.errors = errors
