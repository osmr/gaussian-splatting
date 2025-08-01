from scene.basic_point_cloud import BasicPointCloud
from scene.camera_info import CameraInfo


class SceneInfo:
    def __init__(self,
                 point_cloud: BasicPointCloud,
                 train_cameras: list[CameraInfo],
                 test_cameras: list[CameraInfo],
                 nerf_normalization: dict,
                 ply_path: str,
                 is_nerf_synthetic: bool):
        super(SceneInfo, self).__init__()
        self.point_cloud = point_cloud
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.nerf_normalization = nerf_normalization
        self.ply_path = ply_path
        self.is_nerf_synthetic = is_nerf_synthetic
