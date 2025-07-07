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

from cvutil.random import init_rand
# init_rand(seed=0, forced=True, hard=True)

import os
from typing import Callable, Any
import torch
from random import randint
from gaussian_renderer import network_gui
from gaussian_renderer.render import do_render
import sys
from scene.scene import Scene, GaussianModel
from utils.general_utils import get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments.group_param_parser import GroupParamParser
from arguments.model_params import ModelParams
from arguments.pipline_params import PipelineParams
from arguments.optimization_params import OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from loss.l1_loss import l1_loss
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False
    from loss.ssim_loss import ssim

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

import logging
from cvutil.logger import initialize_logging


def prepare_output_dir(model_path: str):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    logging.info("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)

    return model_path


def prepare_tb_logger(model_path: str,
                      use_tb: bool):
    # Create Tensorboard writer
    tb_writer = None
    if use_tb:
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(model_path)
        else:
            logging.info("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer: Any | None,
                    iteration: int,
                    Ll1: torch.Tensor,
                    loss: torch.Tensor,
                    l1_loss: Callable[..., float],
                    elapsed: float,
                    testing_iterations: list[int],
                    scene: Scene,
                    renderFunc: Callable[..., dict],
                    renderArgs: tuple,
                    train_test_exp: bool):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {
                "name": "test",
                "cameras": scene.get_test_cameras()},
            {
                "name": "train",
                "cameras": [scene.get_train_cameras()[idx % len(scene.get_train_cameras())] for idx in range(5, 30, 5)]
            }
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_file_name),
                            image[None],
                            global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_file_name),
                                gt_image[None],
                                global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logging.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration,
                    config['name'],
                    l1_test,
                    psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def training(model_params: ModelParams,
             pipe_params: PipelineParams,
             opt_params: OptimizationParams,
             testing_iterations: list[int],
             saving_iterations: list[int],
             checkpoint_iterations: list[int],
             checkpoint: str | None,
             debug_from: int,
             use_tb: bool):

    if not SPARSE_ADAM_AVAILABLE and opt_params.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0

    model_params.model_path = prepare_output_dir(model_path=model_params.model_path)
    tb_writer = prepare_tb_logger(
        model_path=model_params.model_path,
        use_tb=use_tb)

    gaussians = GaussianModel(
        sh_degree=model_params.sh_degree,
        optimizer_type=opt_params.optimizer_type)
    scene = Scene(
        gaussians=gaussians,
        model_params=model_params)
    gaussians.training_setup(opt_params=opt_params)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_params)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt_params.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        lr_init=opt_params.depth_l1_weight_init,
        lr_final=opt_params.depth_l1_weight_final,
        max_steps=opt_params.iterations)

    viewpoint_stack = scene.get_train_cameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_loss_l1_depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt_params.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt_params.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe_params.convert_sh_python,
                    pipe_params.compute_cov3d_python,
                    keep_alive,
                    scaling_modifer
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        viewpoint_camera=custom_cam,
                        pc=gaussians,
                        pipe=pipe_params,
                        bg_color=background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=model_params.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(
                        1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, model_params.source_path)
                if do_training and ((iteration < int(opt_params.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.increment_sh_degree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe_params.debug = True

        bg = torch.rand(3, device="cuda") if opt_params.random_background else background

        render_pkg = do_render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe_params,
            bg_color=bg,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=model_params.train_test_exp)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        l1_value = l1_loss(image, gt_image)
        # if False:
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt_params.lambda_dssim) * l1_value + opt_params.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            loss_l1_depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += loss_l1_depth
            loss_l1_depth = loss_l1_depth.item()
        else:
            loss_l1_depth = 0

        loss.backward()

        iter_end.record()

        with ((torch.no_grad())):
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_l1_depth_for_log = 0.4 * loss_l1_depth + 0.6 * ema_loss_l1_depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_loss_l1_depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_params.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer=tb_writer,
                iteration=iteration,
                Ll1=l1_value,
                loss=loss,
                l1_loss=l1_loss,
                elapsed=iter_start.elapsed_time(iter_end),
                testing_iterations=testing_iterations,
                scene=scene,
                renderFunc=do_render,
                renderArgs=(pipe_params, background, 1., SPARSE_ADAM_AVAILABLE, None, model_params.train_test_exp),
                train_test_exp=model_params.train_test_exp)
            if iteration in saving_iterations:
                logging.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt_params.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii_2d[visibility_filter] = torch.max(
                    gaussians.max_radii_2d[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor=viewspace_point_tensor,
                    update_filter=visibility_filter)

                if (iteration > opt_params.densify_from_iter) and (iteration % opt_params.densification_interval == 0):
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        max_grad=opt_params.densify_grad_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent,
                        max_screen_size=size_threshold,
                        radii=radii)

                if (iteration % opt_params.opacity_reset_interval == 0) or (model_params.white_background and (iteration == opt_params.densify_from_iter)):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_params.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                logging.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    model_params = ModelParams()
    pipe_params = PipelineParams()
    opt_params = OptimizationParams()

    parser = ArgumentParser(description="Training script parameters")
    GroupParamParser.export_to_args(
        param_struct=model_params,
        parser=parser,
        name="Model/Dataset Parameters",
        fill_none=False)
    GroupParamParser.export_to_args(
        param_struct=pipe_params,
        parser=parser,
        name="Pipeline Parameters",
        fill_none=False)
    GroupParamParser.export_to_args(
        param_struct=opt_params,
        parser=parser,
        name="Optimization Parameters",
        fill_none=False)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument(
        "--tb",
        action="store_true",
        help="use TensorBoard logging")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="random seed to be fixed")
    parser.add_argument(
        "--log",
        type=str,
        default="train.log",
        help="file name for processing log (relative to the root)")
    parser.add_argument(
        "--log-packages",
        type=str,
        default="cvutil, torch, plyfile, opencv-python, numpy",
        help="list of python packages for logging")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    GroupParamParser.import_from_args(
        param_struct=model_params,
        args=args)
    model_params.source_path = os.path.abspath(model_params.source_path)
    GroupParamParser.import_from_args(
        param_struct=pipe_params,
        args=args)
    GroupParamParser.import_from_args(
        param_struct=opt_params,
        args=args)


    args.seed = init_rand(seed=args.seed, forced=True, hard=True)
    root_dir_path = args.model_path
    logger, _ = initialize_logging(
        logging_dir_path=root_dir_path,
        logging_file_name=args.log,
        main_script_path=__file__,
        script_args=args,
        check_cuda=True)

    logging.info("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(
            wish_host=args.ip,
            wish_port=args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        model_params=model_params,
        pipe_params=pipe_params,
        opt_params=opt_params,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        use_tb=args.tb)

    # All done
    logging.info("\nTraining complete.")
