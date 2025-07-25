import sys
import logging
import torch
from scene.scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render
from gaussian_renderer.render import do_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments.group_param_parser import GroupParamParser
from arguments.model_params import ModelParams
from arguments.pipline_params import PipelineParams
from arguments.utils import get_combined_args
from scene.scene import GaussianModel
from scene.camera import Camera
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path: str,
               name: str,
               iteration: int,
               views: list[Camera],
               gaussians: GaussianModel,
               pipeline: PipelineParams,
               background: torch.Tensor,
               train_test_exp: bool,
               separate_sh: bool):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = do_render(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background,
            separate_sh=separate_sh,
            use_trained_exp=train_test_exp)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        rendering_np = rendering.detach().cpu().numpy()
        np.save(os.path.join(render_path, '{0:05d}'.format(idx) + ".npy"), rendering_np)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams,
                iteration: int,
                pipeline: PipelineParams,
                skip_train: bool,
                skip_test: bool,
                separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            gaussians=gaussians,
            model_params=dataset,
            load_iteration=iteration,
            shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 model_path=dataset.model_path,
                 name="train",
                 iteration=scene.loaded_iter,
                 views=scene.get_train_cameras(),
                 gaussians=gaussians,
                 pipeline=pipeline,
                 background=background,
                 train_test_exp=dataset.train_test_exp,
                 separate_sh=separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.get_test_cameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    model = ModelParams()
    pipeline = PipelineParams()

    parser = ArgumentParser(description="Testing script parameters")
    GroupParamParser.export_to_args(
        param_struct=model,
        parser=parser,
        name="Loading Parameters",
        fill_none=False)
        # fill_none = True)
    GroupParamParser.export_to_args(
        param_struct=pipeline,
        parser=parser,
        name="Pipeline Parameters",
        fill_none=False)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    # args = get_combined_args(parser)

    GroupParamParser.import_from_args(
        param_struct=model,
        args=args)
    GroupParamParser.import_from_args(
        param_struct=pipeline,
        args=args)

    logging.info("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        dataset=model,
        iteration=args.iteration,
        pipeline=pipeline,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        separate_sh=SPARSE_ADAM_AVAILABLE)
