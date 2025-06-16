import numpy as np


class ColmapCameraIntrinsic:
    """
    COLMAP specific camera instrinsic parameters or `camera`.
    """
    def __init__(self,
                 id: int,
                 model_name: str,
                 width: int,
                 height: int,
                 params: np.ndarray):
        super(ColmapCameraIntrinsic, self).__init__()
        assert (model_name == "PINHOLE")

        self.id = id
        self.model_name = model_name
        self.width = width
        self.height = height
        self.params = params
