import logging
import torch
from torch import nn
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2


class Camera(nn.Module):
    def __init__(self,
                 resolution: tuple[int, int],
                 colmap_id: int,
                 R: np.ndarray,
                 T: np.ndarray,
                 fov_x: float,
                 fov_y: float,
                 depth_params: dict | None,
                 image: Image,
                 inv_depth_map: np.ndarray | None,
                 image_name: str,
                 uid: int,
                 trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 scale: float = 1.0,
                 data_device: str = "cuda",
                 train_test_exp: bool = False,
                 is_test_dataset: bool = False,
                 is_test_view: bool = False):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            logging.info(e)
            logging.info(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if inv_depth_map is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(inv_depth_map, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if (depth_params["scale"] < 0.2 * depth_params["med_scale"] or
                        depth_params["scale"] > 5 * depth_params["med_scale"]):
                    self.depth_reliable = False
                    self.depth_mask *= 0

                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(
            R=R,
            t=T,
            translate=trans,
            scale=scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.fov_x,
            fovY=self.fov_y).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
