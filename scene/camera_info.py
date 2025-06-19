import numpy as np


class CameraInfo:
    def __init__(self,
                 uid: int,
                 R: np.ndarray,
                 T: np.ndarray,
                 width: int,
                 height: int,
                 fov_x: np.ndarray,
                 fov_y: np.ndarray,
                 depth_params: dict | None,
                 image_file_path: str,
                 image_file_name: str,
                 depth_path: str,
                 is_test: bool):
        super(CameraInfo, self).__init__()

        self.uid = uid

        self.R = R
        self.T = T

        self.width = width
        self.height = height

        self.fov_x = fov_x
        self.fov_y = fov_y

        self.depth_params = depth_params

        self.image_file_path = image_file_path
        self.image_file_name = image_file_name
        self.depth_path = depth_path

        self.is_test = is_test
