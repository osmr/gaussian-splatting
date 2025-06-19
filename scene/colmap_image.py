import numpy as np


class ColmapImage:
    """
    COLMAP specific camera extrinsic parameters or `image`.
    """
    def __init__(self,
                 image_id: int,
                 qvec: np.ndarray,
                 tvec: np.ndarray,
                 camera_id: int,
                 image_file_name: str):
        super(ColmapImage, self).__init__()
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.image_file_name = image_file_name
        # self.pts2d = pts2d
        # self.pts3d_ids = pts3d_ids
