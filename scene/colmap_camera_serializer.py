import numpy as np
from collections import namedtuple
from scene.colmap_camera import ColmapCamera
from scene.colmap_utils import colmap_binary_read_next_bytes


CameraModel = namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = (
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
)
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}


class ColmapCameraSerializer:

    @staticmethod
    def load_from_txt(txt_file_path: str) -> dict[int, ColmapCamera]:
        """
        Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
        """
        camera_intrinsics = {}
        with open(txt_file_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if (len(line) > 0) and (line[0] != "#"):
                    elements = line.split()
                    camera_id = int(elements[0])
                    model_name = elements[1]
                    width = int(elements[2])
                    height = int(elements[3])
                    params = np.array(tuple(map(float, elements[4:])))
                    camera_intrinsics[camera_id] = ColmapCamera(
                        id=camera_id,
                        model_name=model_name,
                        width=width,
                        height=height,
                        params=params)
        return camera_intrinsics

    @staticmethod
    def load_from_bin(bin_file_path: str) -> dict[int, ColmapCamera]:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        camera_intrinsics = {}
        with open(bin_file_path, "rb") as fid:
            num_cameras = colmap_binary_read_next_bytes(
                fid=fid,
                num_bytes=8,
                format_char_sequence="Q")[0]
            for _ in range(num_cameras):
                camera_properties = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=24,
                    format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=(8 * num_params),
                    format_char_sequence=("d" * num_params))
                camera_intrinsics[camera_id] = ColmapCamera(
                    id=camera_id,
                    model_name=model_name,
                    width=width,
                    height=height,
                    params=np.array(params))
            assert (len(camera_intrinsics) == num_cameras)
        return camera_intrinsics
