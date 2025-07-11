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

import os
import random
import json
import logging
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import extract_scene_info_from_colmap, read_nerf_synthetic_info
from scene.gaussian_model import GaussianModel
from arguments.model_params import ModelParams
from utils.camera_utils import cameraList_from_camInfos, serialize_camera_to_dict
from scene.gaussian_model_serializer import GaussianModelSerializer


class Scene:
    def __init__(self,
                 gaussians: GaussianModel,
                 model_params: ModelParams,
                 load_iteration: int | None = None,
                 shuffle: bool = True,
                 resolution_scales: list[float] = [1.0]):
        self.gaussians = gaussians
        self.model_path = model_params.model_path
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            logging.info("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(model_params.source_path, "sparse")):
            scene_info = extract_scene_info_from_colmap(
                data_dir_path=model_params.source_path,
                images_dir_name=model_params.images,
                depths_dir_name=model_params.depths,
                eval=model_params.eval,
                train_test_exp=model_params.train_test_exp)
        elif os.path.exists(os.path.join(model_params.source_path, "transforms_train.json")):
            logging.info("Found transforms_train.json file, assuming Blender data set!")
            scene_info = read_nerf_synthetic_info(
                path=model_params.source_path,
                white_background=model_params.white_background,
                depths=model_params.depths,
                eval=model_params.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with (open(scene_info.ply_path, "rb") as src_file,
                  open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file):
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(serialize_camera_to_dict(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            logging.info("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                cam_infos=scene_info.train_cameras,
                resolution_scale=resolution_scale,
                model_params=model_params,
                is_nerf_synthetic=scene_info.is_nerf_synthetic,
                is_test_dataset=False)
            logging.info("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                cam_infos=scene_info.test_cameras,
                resolution_scale=resolution_scale,
                model_params=model_params,
                is_nerf_synthetic=scene_info.is_nerf_synthetic,
                is_test_dataset=True)

        if self.loaded_iter:
            GaussianModelSerializer.load_ply(
                model=self.gaussians,
                path=self._get_point_cloud_ply_file_path(self.loaded_iter),
                use_train_test_exp=model_params.train_test_exp)
        else:
            self.gaussians.create_from_pcd(
                pcd=scene_info.point_cloud,
                cam_infos=scene_info.train_cameras,
                spatial_lr_scale=self.cameras_extent)

    def _get_point_cloud_ply_file_path(self,
                                       iteration: int):
        return os.path.join(self.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")

    def save(self,
             iteration: int):
        GaussianModelSerializer.save_ply(
            model=self.gaussians,
            path=self._get_point_cloud_ply_file_path(iteration)
        )

        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def get_train_cameras(self,
                          scale: float = 1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self,
                         scale: float = 1.0):
        return self.test_cameras[scale]
