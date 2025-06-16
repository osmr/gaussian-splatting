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

import logging
from scene.camera import Camera
import numpy as np
from arguments.model_params import ModelParams
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False


def loadCam(args: ModelParams,
            id: int,
            cam_info: CameraInfo,
            resolution_scale: float,
            is_nerf_synthetic: bool,
            is_test_dataset: bool):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                inv_depth_map = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                inv_depth_map = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            logging.info(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            logging.info(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            logging.info(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        inv_depth_map = None

    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = (round(orig_w / (resolution_scale * args.resolution)),
                      round(orig_h / (resolution_scale * args.resolution)))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    logging.info(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(
        resolution=resolution,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        fov_x=cam_info.fov_x,
        fov_y=cam_info.fov_y,
        depth_params=cam_info.depth_params,
        image=image,
        inv_depth_map=inv_depth_map,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        train_test_exp=args.train_test_exp,
        is_test_dataset=is_test_dataset,
        is_test_view=cam_info.is_test)


def cameraList_from_camInfos(cam_infos: list[CameraInfo],
                             resolution_scale: float,
                             model_params: ModelParams,
                             is_nerf_synthetic: bool,
                             is_test_dataset: bool):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(
            args=model_params,
            id=id,
            cam_info=c,
            resolution_scale=resolution_scale,
            is_nerf_synthetic=is_nerf_synthetic,
            is_test_dataset=is_test_dataset))

    return camera_list


def camera_to_JSON(id: int,
                   camera: CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.fov_y, camera.height),
        "fx": fov2focal(camera.fov_x, camera.width)
    }
    return camera_entry
