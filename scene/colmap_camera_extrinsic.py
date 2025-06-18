import numpy as np


class ColmapCameraExtrinsic:
    """
    COLMAP specific camera extrinsic parameters or `image`.
    """
    def __init__(self,
                 image_id: int,
                 qvec: np.ndarray,
                 tvec: np.ndarray,
                 camera_id: int,
                 image_file_name: str,
                 pts2d: np.ndarray,
                 pts3d_ids: np.ndarray):
        super(ColmapCameraExtrinsic, self).__init__()
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.image_file_name = image_file_name
        self.pts2d = pts2d
        self.pts3d_ids = pts3d_ids
