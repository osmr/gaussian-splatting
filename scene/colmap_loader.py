"""
    # Copyright (C) 2023, Inria
    # GRAPHDECO research group, https://team.inria.fr/graphdeco
    # All rights reserved.
    #
    # This software is free for non-commercial, research and evaluation use
    # under the terms of the LICENSE.md file.
    #
    # For inquiries contact  george.drettakis@inria.fr
"""

import numpy as np
# import collections


# CameraModel = collections.namedtuple(
#     "CameraModel", ["model_id", "model_name", "num_params"])
# Camera = collections.namedtuple(
#     "Camera", ["id", "model", "width", "height", "params"])
# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# Point3D = collections.namedtuple(
#     "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
# CAMERA_MODELS = {
#     CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
#     CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
#     CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
#     CameraModel(model_id=3, model_name="RADIAL", num_params=5),
#     CameraModel(model_id=4, model_name="OPENCV", num_params=8),
#     CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
#     CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
#     CameraModel(model_id=7, model_name="FOV", num_params=5),
#     CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
#     CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
#     CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
# }
# CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
#                          for camera_model in CAMERA_MODELS])
# CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
#                            for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# def read_colmap_bin_array(path):
#     """
#     Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py
#
#     :param path: path to the colmap binary file.
#     :return: nd array with the floating point values in the value
#     """
#     with open(path, "rb") as fid:
#         width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
#                                                 usecols=(0, 1, 2), dtype=int)
#         fid.seek(0)
#         num_delimiter = 0
#         byte = fid.read(1)
#         while True:
#             if byte == b"&":
#                 num_delimiter += 1
#                 if num_delimiter >= 3:
#                     break
#             byte = fid.read(1)
#         array = np.fromfile(fid, np.float32)
#     array = array.reshape((width, height, channels), order="F")
#     return np.transpose(array, (1, 0, 2)).squeeze()
