#!/usr/bin/env python3

import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from environment import Environment
from model import Network, init_func
from utils import read_img


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def train(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    torch.random.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    with torch.no_grad():
        valid_image_extensions = tuple(Image.registered_extensions().keys())
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
        num_shapes = len(paths) + 1  # One extra for null shape
        torch.save(
            shapes, os.path.join(HydraConfig.get().runtime.output_dir, "shapes.pth")
        )

    model = Network(
        num_shapes=num_shapes,
        param_per_stroke=cfg.model.param_per_stroke,
        num_strokes=cfg.num_strokes,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_enc_layers=cfg.model.n_enc_layers,
        n_dec_layers=cfg.model.n_dec_layers,
    )
    model.apply(init_func)

    model.to(device)

    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, **cfg.optimizer.kwargs
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer.type} not implemented")

    environment = Environment(
        shapes=shapes,
        d=cfg.param_shape,
        patch_size=cfg.patch_size,
        num_strokes=cfg.num_strokes,
        device=device,
        batch_size=cfg.batch_size,
        lambda_gt=cfg.lambda_gt,
        lambda_idx=cfg.lambda_idx,
        lambda_pixel=cfg.lambda_pixel,
        lambda_w=cfg.lambda_w,
    )

    writer = SummaryWriter(os.path.join(HydraConfig.get().runtime.output_dir, "logs"))

    progress = tqdm(range(cfg.total_iters))
    for i in progress:
        try:
            optimizer.zero_grad()
            render, old = environment.reset()
            param, idx = model(render, old)
            recreation, loss = environment.step(param, idx)
            loss.backward()
            optimizer.step()

            if i % cfg.log_interval == 0:
                loss_pixel, loss_gt, loss_w, loss_idx = environment.get_losses()

                progress.set_postfix(
                    pixel_loss=f"{loss_pixel:.4f}",
                    param_loss=f"{loss_gt:.4f}",
                    w_loss=f"{loss_w:.4f}",
                    idx_loss=f"{loss_idx:.4f}",
                )

                writer.add_scalar("Loss/pixel_loss", loss_pixel, i)
                writer.add_scalar("Loss/param_loss", loss_gt, i)
                writer.add_scalar("Loss/idx_loss", loss_idx, i)
                writer.add_scalar("Loss/w_loss", loss_w, i)

            if i % cfg.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        HydraConfig.get().runtime.output_dir,
                        f"model_{str(i).zfill(8)}.pth",
                    ),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        HydraConfig.get().runtime.output_dir,
                        f"optimizer_{str(i).zfill(8)}.pth",
                    ),
                )

                with torch.no_grad():
                    render, recreation = make_grid(render), make_grid(recreation)
                    writer.add_image("Render", render, i)
                    writer.add_image("Reconstruction", recreation, i)

        except KeyboardInterrupt:
            break

        except Exception as e:
            continue

    torch.save(
        model.state_dict(),
        os.path.join(HydraConfig.get().runtime.output_dir, "model_final.pth"),
    )


if __name__ == "__main__":
    train()
