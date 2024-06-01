#!/usr/bin/env python3

import math
import os
from typing import Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from environment import Environment
from model import Network
from utils import crop, pad, param2img_parallel, read_img


def inference(
    cfg: Union[OmegaConf, DictConfig, ListConfig],
    model: nn.Module,
    shapes: torch.Tensor,
):
    device = torch.device(cfg.device)

    with torch.no_grad():
        num_shapes = shapes.shape[0]

        original_image = read_img(cfg.target_path)[:, :3, ...].to(device)
        original_image = F.interpolate(original_image, scale_factor=cfg.scale)
        original_h, original_w = original_image.shape[-2:]
        K = max(math.ceil(math.log2(max(original_h, original_w) / cfg.patch_size)), 0)
        original_img_pad_size = cfg.patch_size * (2**K)
        original_img_pad = pad(
            original_image, original_img_pad_size, original_img_pad_size
        )
        final_result = torch.zeros_like(original_img_pad).to(device)

    model.to(device)

    environment = Environment(
        shapes=shapes,
        d=cfg.param_shape,
        patch_size=cfg.patch_size,
        num_strokes=cfg.num_strokes,
        device=device,
        batch_size=1,
        train=False,
    )

    with torch.no_grad():
        for layer in tqdm(range(0, K + 1)):
            layer_size = cfg.patch_size * (2**layer)
            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(
                final_result,
                (cfg.patch_size * (2**layer), cfg.patch_size * (2**layer)),
            )
            img_patch = F.unfold(
                img,
                (cfg.patch_size, cfg.patch_size),
                stride=(cfg.patch_size, cfg.patch_size),
            )
            result_patch = F.unfold(
                result,
                (cfg.patch_size, cfg.patch_size),
                stride=(cfg.patch_size, cfg.patch_size),
            )
            patch_num = (layer_size - cfg.patch_size) // cfg.patch_size + 1

            img_patch = (
                img_patch.permute(0, 2, 1)
                .contiguous()
                .view(-1, 3, cfg.patch_size, cfg.patch_size)
                .contiguous()
            )
            result_patch = (
                result_patch.permute(0, 2, 1)
                .contiguous()
                .view(-1, 3, cfg.patch_size, cfg.patch_size)
                .contiguous()
            )
            shape_param, stroke_idx = model(img_patch, result_patch)
            stroke_idx = stroke_idx.argmax(dim=-1)
            stroke_decision = stroke_idx != num_shapes - 1

            param = shape_param.view(
                1, patch_num, patch_num, cfg.num_strokes, 4
            ).contiguous()
            idx = stroke_idx.view(
                1, patch_num, patch_num, cfg.num_strokes, 1
            ).contiguous()
            decision = (
                stroke_decision.view(1, patch_num, patch_num, cfg.num_strokes)
                .contiguous()
                .bool()
            )
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2
            final_result = param2img_parallel(
                param, idx, decision, final_result, environment
            )

        border_size = original_img_pad_size // (2 * patch_num)
        img = F.interpolate(
            original_img_pad,
            (cfg.patch_size * (2**layer), cfg.patch_size * (2**layer)),
        )
        result = F.interpolate(
            final_result, (cfg.patch_size * (2**layer), cfg.patch_size * (2**layer))
        )
        img = F.pad(
            img,
            [
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                0,
                0,
                0,
                0,
            ],
        )
        result = F.pad(
            result,
            [
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                cfg.patch_size // 2,
                0,
                0,
                0,
                0,
            ],
        )
        img_patch = F.unfold(
            img,
            (cfg.patch_size, cfg.patch_size),
            stride=(cfg.patch_size, cfg.patch_size),
        )
        result_patch = F.unfold(
            result,
            (cfg.patch_size, cfg.patch_size),
            stride=(cfg.patch_size, cfg.patch_size),
        )
        final_result = F.pad(
            final_result,
            [border_size, border_size, border_size, border_size, 0, 0, 0, 0],
        )
        h = (img.shape[2] - cfg.patch_size) // cfg.patch_size + 1
        w = (img.shape[3] - cfg.patch_size) // cfg.patch_size + 1
        img_patch = (
            img_patch.permute(0, 2, 1)
            .contiguous()
            .view(-1, 3, cfg.patch_size, cfg.patch_size)
            .contiguous()
        )
        result_patch = (
            result_patch.permute(0, 2, 1)
            .contiguous()
            .view(-1, 3, cfg.patch_size, cfg.patch_size)
            .contiguous()
        )
        shape_param, stroke_idx = model(img_patch, result_patch)
        stroke_idx = stroke_idx.argmax(dim=-1)
        stroke_decision = stroke_idx != num_shapes - 1

        param = shape_param.view(1, h, w, cfg.num_strokes, 4).contiguous()
        idx = stroke_idx.view(1, h, w, cfg.num_strokes, 1).contiguous()
        decision = stroke_decision.view(1, h, w, cfg.num_strokes).contiguous().bool()
        param[..., :2] = param[..., :2] / 2 + 0.25
        param[..., 2:4] = param[..., 2:4] / 2
        final_result = param2img_parallel(
            param, idx, decision, final_result, environment
        )

        # Center crop the final result to the original image size
        final_result = crop(final_result, original_h, original_w)
        save_image(final_result[0], cfg.recreation_path)


@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def cli_inference(cfg: Union[OmegaConf, DictConfig, ListConfig]):
    print("Current working dir:", os.getcwd())

    model = Network(
        num_shapes=cfg.num_shapes,
        param_per_stroke=cfg.param_per_stroke,
        num_strokes=cfg.num_strokes,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        n_dec_layers=cfg.n_dec_layers,
    )
    try:
        model.load_state_dict(torch.load(cfg.weights_path))
    except:
        model.load_state_dict(torch.load(cfg.weights_path)["model"])

    valid_image_extensions = tuple(Image.registered_extensions().keys())

    if cfg.shapes_path.endswith(".pth"):
        shapes = torch.load(cfg.shapes_path)

    else:
        paths = sorted(
            [
                os.path.join(cfg.shapes_path, f)
                for f in os.listdir(cfg.shapes_path)
                if f.endswith(valid_image_extensions)
            ]
        )
        shapes = torch.cat([read_img(path) for path in paths])
        zeros_image = torch.zeros_like(shapes[-1])
        shapes = torch.cat([shapes, zeros_image.unsqueeze(0)])
        shapes = F.interpolate(
            shapes, (2 * cfg.patch_size, 2 * cfg.patch_size), mode="bilinear"
        )

    inference(cfg, model, shapes)


if __name__ == "__main__":
    cli_inference()
