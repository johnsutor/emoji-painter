#!/usr/bin/env python3

import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils import gaussian_w_distance


class Environment:
    old: torch.Tensor
    render: torch.Tensor
    rec: torch.Tensor
    gt_idx: torch.Tensor
    gt_param: torch.Tensor
    gt_decision: torch.Tensor
    """Environment representing the painter and it's parameters"""

    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        shapes: torch.Tensor,
        d: int,
        patch_size: int,
        num_strokes: int,
        lambda_gt: float = 1.0,
        lambda_idx: float = 1.0,
        lambda_pixel: float = 10.0,
        lambda_w: float = 10.0,
    ):
        """
        Args:
            device (torch.device): Device to run the environment on
            batch_size (int): Batch size
            shapes (torch.Tensor): Shapes to use for rendering
            d (int): Number of parameters per shape
            patch_size (int): Size of the patch
            num_strokes (int): Number of strokes to use
        """
        self.device = device
        self.batch_size = batch_size
        self.shapes = shapes.to(self.device)
        self.num_shapes = shapes.shape[0]
        self.d = d
        self.patch_size = patch_size
        self.num_strokes = num_strokes
        self.lambda_gt = lambda_gt
        self.lambda_idx = lambda_idx
        self.lambda_pixel = lambda_pixel
        self.lambda_w = lambda_w

    def param2stroke(self, param: torch.Tensor, idx: torch.Tensor, H: int, W: int):
        """Convert parameters to a stroke using grid sampling

        Args:
            param (torch.Tensor): Parameters of the stroke
            idx (torch.Tensor): Index of the stroke
            H (int): Height of the image
            W (int): Width of the image

        Returns:
            (torch.Tensor, torch.Tensor): Brush and alphas of the stroke
        """
        b = max(param.shape[0], 1)
        x0, y0, w, theta = param.split(1, dim=1)

        # If idx is integers, simply get the idx. Otherwise,
        # Call the gumbel softmax (treating idx like logits)
        # and sample, similar to Transformer
        if idx.dtype == torch.int64 or idx.dtype == torch.long:
            idx = idx.view(-1)
            images = self.shapes[idx]
        else:
            images = (
                F.gumbel_softmax(idx, hard=True).view(  # tau=0.5
                    -1, self.num_shapes, 1, 1, 1
                )
                * self.shapes
            ).sum(dim=1)

        h = w
        brush, alphas = images.split((3, 1), dim=1)
        sin_theta = torch.sin(math.pi * theta)
        cos_theta = torch.cos(math.pi * theta)

        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)

        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1).squeeze(-1)
        grid = torch.nn.functional.affine_grid(
            warp, torch.Size((b, 3, H, W)), align_corners=False
        )
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        return brush, alphas

    def get_losses(self):
        """Get the individual losses

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): pixel, param, wasserstein, and index losses (in that order)
        """
        return self.loss_pixel, self.loss_gt, self.loss_w, self.loss_idx

    def reset(self):
        """Creates the training canvas

        Returns:

        """
        with torch.no_grad():
            self.old = torch.zeros(
                self.batch_size, 3, self.patch_size, self.patch_size
            ).to(self.device)

            # Generate starter canvas
            start_idx = torch.randint(
                0, self.num_shapes, (self.batch_size * self.num_strokes,)
            ).to(self.device)
            start_params = torch.rand((self.batch_size * self.num_strokes, self.d)).to(
                self.device
            )

            brush, alphas = self.param2stroke(
                start_params, start_idx, self.patch_size, self.patch_size
            )
            brush, alphas = (
                brush.reshape(
                    self.batch_size,
                    self.num_strokes,
                    3,
                    self.patch_size,
                    self.patch_size,
                ),
                alphas.reshape(
                    self.batch_size,
                    self.num_strokes,
                    1,
                    self.patch_size,
                    self.patch_size,
                ),
            )

            for i in range(self.num_strokes):
                self.old = brush[:, i, ...] * alphas[:, i, ...] + self.old * (
                    1 - alphas[:, i, ...]
                )

            self.render = self.old.clone()

            # Generate the actual parameters
            self.gt_idx = torch.randint(
                0, self.num_shapes, (self.batch_size * self.num_strokes,)
            ).to(self.device)
            self.gt_param = (
                torch.rand((self.batch_size * self.num_strokes, self.d)).to(self.device)
                * 0.8
                + 0.1
            )

            brush, alphas = self.param2stroke(
                self.gt_param, self.gt_idx, self.patch_size, self.patch_size
            )
            brush, alphas = (
                brush.reshape(
                    self.batch_size,
                    self.num_strokes,
                    3,
                    self.patch_size,
                    self.patch_size,
                ),
                alphas.reshape(
                    self.batch_size,
                    self.num_strokes,
                    1,
                    self.patch_size,
                    self.patch_size,
                ),
            )
            self.gt_idx = self.gt_idx.reshape(self.batch_size, self.num_strokes)
            self.gt_param = self.gt_param.reshape(
                self.batch_size, self.num_strokes, self.d
            )

            # Iterate over, determine if it's covered by latter images
            overlaps = torch.zeros((self.batch_size, self.num_strokes))

            # Last index will never be covered, don't need check.
            # Reversed so that we don't have to recompute all subsequent
            # masks at each step
            mask = torch.zeros((self.batch_size, self.patch_size, self.patch_size)).to(
                self.device
            )

            for index in reversed(range(1, self.num_strokes)):
                mask = torch.logical_or(mask, alphas[:, index, ...])

                # Overlap using logical and
                current_sum = torch.sum(alphas[:, index - 1, ...], dim=(1, 2, 3))
                overlap = (
                    torch.sum(
                        torch.logical_and(alphas[:, index - 1, ...], mask),
                        dim=(1, 2, 3),
                    )
                    / current_sum
                )
                overlap = torch.nan_to_num(overlap, 1.0)

                overlaps[:, index - 1] = overlap

            # Determine the amount of overlap, dump those with too much
            self.gt_decision = (1 - overlaps.clamp_(0, 1)).to(self.device) > 0.6
            float_decisions = self.gt_decision.float().view(
                self.batch_size, self.num_strokes, 1, 1, 1
            )

            for i in range(self.num_strokes):
                self.render = brush[:, i, ...] * alphas[:, i, ...] * float_decisions[
                    :, i, ...
                ] + self.render * (1 - alphas[:, i, ...] * float_decisions[:, i, ...])

        return self.render, self.old

    def calculate_loss(self):
        """Calculate the loss of the current prediction

        Returns:
            torch.Tensor: Loss of the current prediction
        """
        self.loss_pixel = F.l1_loss(self.rec, self.render) * self.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):
                is_valid_gt = self.gt_decision[i].bool()
                valid_gt_param = self.gt_param[i, is_valid_gt]

                cost_matrix_dist = torch.cdist(
                    self.pred_param[i], valid_gt_param, p=2
                ).pow(2)  # torch.cdist(self.pred_param[i], valid_gt_param, p=1)
                pred_param_broad = (
                    self.pred_param[i]
                    .unsqueeze(1)
                    .contiguous()
                    .repeat(1, valid_gt_param.shape[0], 1)
                )
                valid_gt_param_broad = (
                    valid_gt_param.unsqueeze(0)
                    .contiguous()
                    .repeat(self.pred_param.shape[1], 1, 1)
                )
                cost_matrix_w = gaussian_w_distance(
                    pred_param_broad, valid_gt_param_broad
                )
                idx = self.pred_idx[i]
                cost_matrix_idx = (
                    F.cross_entropy(idx, self.gt_idx[i], reduction="none")
                    .unsqueeze(-1)
                    .repeat(1, valid_gt_param.shape[0])
                )
                r, c = linear_sum_assignment(
                    (cost_matrix_dist + cost_matrix_idx + cost_matrix_w).cpu()
                )

                r_idx.append(
                    torch.tensor(r + self.pred_param.shape[1] * i, device=self.device)
                )
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
        r_idx, c_idx = r_idx.to(self.device), c_idx.to(self.device)
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_valid_gt_idx = self.gt_idx[self.gt_decision.bool(), ...]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_idx = self.pred_idx.view(-1, self.pred_idx.shape[2]).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        paired_gt_idx = all_valid_gt_idx[c_idx]
        paired_pred_idx = all_pred_idx[r_idx, :]
        self.loss_gt = F.l1_loss(paired_pred_param, paired_gt_param) * self.lambda_gt
        self.loss_w = (
            gaussian_w_distance(paired_pred_param, paired_gt_param).mean()
            * self.lambda_w
        )
        self.loss_idx = (
            F.cross_entropy(paired_pred_idx, paired_gt_idx) * self.lambda_idx
        )
        loss = self.loss_pixel + self.loss_gt + self.loss_w + self.loss_idx
        return loss

    def step(self, param, idx):
        self.pred_idx = idx
        self.pred_decision = idx.argmax(dim=-1) == self.num_shapes - 1
        self.pred_param = param
        foregrounds, alphas = self.param2stroke(
            param.reshape(-1, self.d),
            idx.reshape(-1, self.num_shapes),
            self.patch_size,
            self.patch_size,
        )

        foregrounds = foregrounds.view(
            -1, self.num_strokes, 3, self.patch_size, self.patch_size
        )
        alphas = alphas.view(-1, self.num_strokes, 1, self.patch_size, self.patch_size)

        self.rec = self.old.clone()
        for j in range(foregrounds.shape[1]):
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            self.rec = foreground * alpha + self.rec * (1 - alpha)

        loss = self.calculate_loss()
        return self.rec, loss
