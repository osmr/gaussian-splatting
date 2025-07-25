import sys
import logging
import torch
import numpy as np
from scene.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.render import do_render
import torchvision
from argparse import ArgumentParser
from arguments.group_param_parser import GroupParamParser
from arguments.model_params import ModelParams
from arguments.pipline_params import PipelineParams
from scene.scene import GaussianModel
from scene.camera import Camera


def render_view0(dataset: ModelParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            gaussians=gaussians,
            model_params=dataset,
            load_iteration=-1,
            shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.get_train_cameras() 
        view = views[0]

        pipeline = PipelineParams()
        rendering = do_render(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background,
            separate_sh=False,
            use_trained_exp=dataset.train_test_exp)["render"]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        model_path = dataset.model_path
        name = "train"
        iteration = scene.loaded_iter
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        idx = 0
        # rendering_gt = torchvision.io.read_image(path=os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_gt_np = np.load(os.path.join(render_path, '{0:05d}'.format(idx) + ".npy"))
        rendering_gt = torch.from_numpy(rendering_gt_np).cuda()
        # rendering_gt = torchvision.io.read_image(path=os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        is_equal = torch.equal(rendering, rendering_gt)
        logging.info("is_equal={}".format(is_equal))
        pass


if __name__ == "__main__":
    model = ModelParams()

    parser = ArgumentParser(description="Testing script parameters")
    GroupParamParser.export_to_args(
        param_struct=model,
        parser=parser,
        name="Loading Parameters",
        fill_none=False)

    args = parser.parse_args(sys.argv[1:])

    GroupParamParser.import_from_args(
        param_struct=model,
        args=args)

    render_view0(dataset=model)
