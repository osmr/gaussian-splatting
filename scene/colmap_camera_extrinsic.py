import numpy as np


class ColmapCameraExtrinsic:
    """
    COLMAP specific camera extrinsic parameters or `image`.
    """
    def __init__(self,
                 id: int,
                 qvec: np.ndarray,
                 tvec: np.ndarray,
                 camera_id: int,
                 image_name: str,
                 xys: np.ndarray,
                 point3d_ids: np.ndarray):
        super(ColmapCameraExtrinsic, self).__init__()
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.image_name = image_name
        self.xys = xys
        self.point3d_ids = point3d_ids
