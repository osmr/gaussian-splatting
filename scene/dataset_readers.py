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
import sys
import logging
from PIL import Image
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.sh_utils import SH2RGB
from scene.basic_point_cloud import BasicPointCloud
from scene.basic_point_cloud_serializer import BasicPointCloudSerializer
from scene.colmap_point_cloud_serializer import ColmapPointCloudSerializer
from scene.colmap_camera_serializer import ColmapCameraSerializer
from scene.colmap_image_serializer import ColmapImageSerializer
from scene.camera_info import CameraInfo
from scene.scene_info import SceneInfo



def get_nerf_pp_norm(cam_infos: list[CameraInfo]) -> dict:
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def create_camera_infos_from_colmap_data(colmap_cam_extrinsics: dict,
                                         colmap_cam_intrinsics: dict,
                                         depths_params: dict | None,
                                         images_folder: str,
                                         depths_folder: str | None,
                                         test_cam_names_list: list[str]) -> list[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(colmap_cam_extrinsics):
        sys.stdout.write("\r")
        # logging.info('\r')

        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(colmap_cam_extrinsics)))
        sys.stdout.flush()
        # logging.info("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))

        extrinsics = colmap_cam_extrinsics[key]
        intrinsics = colmap_cam_intrinsics[extrinsics.camera_id]
        height = intrinsics.height
        width = intrinsics.width

        uid = intrinsics.id
        R = np.transpose(qvec2rotmat(extrinsics.qvec))
        T = np.array(extrinsics.tvec)

        if intrinsics.model_name == "SIMPLE_PINHOLE":
            focal_length_x = intrinsics.params[0]
            fov_x = focal2fov(focal_length_x, width)
            fov_y = focal2fov(focal_length_x, height)
        elif intrinsics.model_name == "PINHOLE":
            focal_length_x = intrinsics.params[0]
            focal_length_y = intrinsics.params[1]
            fov_x = focal2fov(focal_length_x, width)
            fov_y = focal2fov(focal_length_y, height)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extrinsics.image_file_name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extrinsics.name[:-n_remove]]
            except Exception:
                logging.info("\n{} not found in depths_params".format(key))

        image_file_path = os.path.join(images_folder, extrinsics.image_file_name)
        image_file_name = extrinsics.image_file_name
        depth_path = os.path.join(depths_folder, f"{extrinsics.image_file_name[:-n_remove]}.png") if depths_folder else ""

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            width=width,
            height=height,
            fov_x=fov_x,
            fov_y=fov_y,
            depth_params=depth_params,
            image_file_path=str(image_file_path),
            image_file_name=image_file_name,
            depth_path=depth_path,
            is_test=image_file_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    # logging.info('\n')
    return cam_infos


def extract_scene_info_from_colmap(data_dir_path: str,
                                   images_dir_name: str,
                                   depths_dir_name: str,
                                   eval: bool,
                                   train_test_exp: bool,
                                   llff_hold: int = 8):
    colmap_metainfo_files_dir_path = os.path.join(data_dir_path, "sparse", "0")

    colmap_images_file_stem_name = "images"
    colmap_cameras_file_stem_name = "cameras"
    colmap_images_file_stem_path = os.path.join(colmap_metainfo_files_dir_path, colmap_images_file_stem_name)
    colmap_cameras_file_stem_path = os.path.join(colmap_metainfo_files_dir_path, colmap_cameras_file_stem_name)
    colmap_bin_file_ext = "bin"
    colmap_txt_file_ext = "txt"

    colmap_images_bin_file_path = "{}.{}".format(colmap_images_file_stem_path, colmap_bin_file_ext)
    if os.path.exists(colmap_images_bin_file_path):
        colmap_cameras_bin_file_path = "{}.{}".format(colmap_cameras_file_stem_path, colmap_bin_file_ext)
        colmap_shots = ColmapImageSerializer.load_from_bin(bin_file_path=colmap_images_bin_file_path)
        colmap_cameras = ColmapCameraSerializer.load_from_bin(bin_file_path=colmap_cameras_bin_file_path)
    else:
        colmap_images_txt_file_path = "{}.{}".format(colmap_images_file_stem_path, colmap_txt_file_ext)
        colmap_cameras_txt_file_path = "{}.{}".format(colmap_cameras_file_stem_path, colmap_txt_file_ext)
        colmap_shots = ColmapImageSerializer.load_from_txt(txt_file_path=colmap_images_txt_file_path)
        colmap_cameras = ColmapCameraSerializer.load_from_txt(txt_file_path=colmap_cameras_txt_file_path)

    depth_params_file_path = os.path.join(colmap_metainfo_files_dir_path, "depth_params.json")

    depths_params = None
    if depths_dir_name:
        try:
            with open(depth_params_file_path, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            logging.info(f"Error: depth_params.json file not found at path '{depth_params_file_path}'.")
            sys.exit(1)
        except Exception as e:
            logging.info(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in data_dir_path:
            llff_hold = 8
        if llff_hold:
            logging.info("------------LLFF HOLD-------------")
            cam_names = [colmap_shots[cam_id].name for cam_id in colmap_shots]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llff_hold == 0]
        else:
            with open(os.path.join(colmap_metainfo_files_dir_path, "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images_dir_name is None else images_dir_name
    cam_infos_unsorted = create_camera_infos_from_colmap_data(
        colmap_cam_extrinsics=colmap_shots,
        colmap_cam_intrinsics=colmap_cameras,
        depths_params=depths_params,
        images_folder=os.path.join(data_dir_path, reading_dir),
        depths_folder=os.path.join(data_dir_path, depths_dir_name) if depths_dir_name else None,
        test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_file_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    pcd_file_stem_path = os.path.join(colmap_metainfo_files_dir_path, "points3D")
    pcd_ply_file_path = "{}.{}".format(pcd_file_stem_path, "ply")
    pcd_bin_file_path = "{}.{}".format(pcd_file_stem_path, "bin")
    pcd_txt_file_path = "{}.{}".format(pcd_file_stem_path, "txt")
    if not os.path.exists(pcd_ply_file_path):
        logging.info("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            # xyz, rgb, _ = read_points3D_binary(pcd_bin_file_path)
            cm_pcd = ColmapPointCloudSerializer.load_from_bin(bin_file_path=pcd_bin_file_path)
        except Exception:
            # xyz, rgb, _ = read_points3D_text(pcd_txt_file_path)
            cm_pcd = ColmapPointCloudSerializer.load_from_txt(txt_file_path=pcd_txt_file_path)

        BasicPointCloudSerializer.save_to_ply(
            data=BasicPointCloud(
                positions=cm_pcd.positions,
                colors=(cm_pcd.colors / 255.0)),
            ply_file_path=pcd_ply_file_path)
    try:
        pcd = BasicPointCloudSerializer.load_from_ply(ply_file_path=pcd_ply_file_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=pcd_ply_file_path,
        is_nerf_synthetic=False)
    return scene_info


def read_cameras_from_transforms(path,
                                 transformsfile,
                                 depths_folder,
                                 white_background,
                                 is_test,
                                 extension: str = ".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(
                uid=idx,
                R=R,
                T=T,
                fov_x=FovX,
                fov_y=FovY,
                image_file_path=str(image_path),
                image_file_name=image_name,
                width=image.size[0],
                height=image.size[1],
                depth_path=depth_path,
                depth_params=None,
                is_test=is_test))

    return cam_infos


def read_nerf_synthetic_info(path: str,
                             white_background: bool,
                             depths: str,
                             eval: bool,
                             extension: str = ".png"):

    depths_folder = os.path.join(path, depths) if depths != "" else ""
    logging.info("Reading Training Transforms")
    train_cam_infos = read_cameras_from_transforms(
        path=path,
        transformsfile="transforms_train.json",
        depths_folder=depths_folder,
        white_background=white_background,
        is_test=False,
        extension=extension)
    logging.info("Reading Test Transforms")
    test_cam_infos = read_cameras_from_transforms(
        path=path,
        transformsfile="transforms_test.json",
        depths_folder=depths_folder,
        white_background=white_background,
        is_test=True,
        extension=extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    pcd_ply_file_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(pcd_ply_file_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        logging.info(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        BasicPointCloudSerializer.save_to_ply(
            data=BasicPointCloud(
                positions=xyz,
                colors=SH2RGB(shs)),
            ply_file_path=pcd_ply_file_path)
    try:
        pcd = BasicPointCloudSerializer.load_from_ply(ply_file_path=pcd_ply_file_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=pcd_ply_file_path,
        is_nerf_synthetic=True)
    return scene_info
