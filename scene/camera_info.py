from typing import NamedTuple
import numpy as np


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    depth_params: dict | None
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
