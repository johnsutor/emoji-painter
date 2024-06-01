#!/usr/bin/env python3

import math
import os
from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


def flatten_dict(nested_dict, parent_key="", separator="."):
    """
    Flatten a nested dictionary, where the keys are concatenated with a separator.

    Args:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str, optional): The parent key for the current level. Defaults to ''.
        separator (str, optional): The separator to use when concatenating keys. Defaults to '.'.

    Returns:
        dict: The flattened dictionary.
    """
    flattened_dict = {}

    for key, value in nested_dict.items():
        value = str(value)
        new_key = parent_key + separator + key if parent_key else key

        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, separator))
        else:
            flattened_dict[new_key] = value

    return flattened_dict


def read_img(img_path: os.PathLike):
    """
    Helper for reading in an image from a path and converting it to a PyTorch tensor.

    Args:
        img_path (os.PathLike): The path to the image to read.

    Returns:
        torch.Tensor: The image as a PyTorch tensor.
    """
    img = Image.open(img_path).convert("RGBA")
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    return img


def get_sigma_sqrt(w: torch.Tensor, theta: torch.Tensor):
    """Calculate the square root of the covariance matrix.

    Args:
        w (torch.Tensor): The weight parameter.
        theta (torch.Tensor): The rotation parameter.

    Returns:
        torch.Tensor: The square root of the covariance matrix.
    """
    sigma_00 = w * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
    sigma_01 = (w - w) * torch.cos(theta) * torch.sin(theta) / 2
    sigma_11 = w * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
    sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
    sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
    sigma = torch.stack([sigma_0, sigma_1], dim=-2)
    return sigma


def get_sigma(w: torch.Tensor, theta: torch.Tensor):
    """Calculate the covariance matrix.

    Args:
        w (torch.Tensor): The weight parameter.
        theta (torch.Tensor): The rotation parameter.

    Returns:
        torch.Tensor: The covariance matrix.
    """
    sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
    sigma_01 = (w * w - w * w) * torch.cos(theta) * torch.sin(theta) / 4
    sigma_11 = w * w * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
    sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
    sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
    sigma = torch.stack([sigma_0, sigma_1], dim=-2)
    return sigma


def gaussian_w_distance(param_1: torch.Tensor, param_2: torch.Tensor):
    """Calculates the Wasserstein distance between two Gaussian distributions.

    Args:
        param_1 (torch.Tensor): The parameters of the first Gaussian distribution.
        param_2 (torch.Tensor): The parameters of the second Gaussian distribution.

    Returns:
        torch.Tensor: The Wasserstein distance between the two distributions.
    """
    mu_1, w_1, theta_1 = torch.split(param_1, (2, 1, 1), dim=-1)
    w_1 = w_1.squeeze(-1)
    theta_1 = torch.acos(torch.tensor(-1.0, device=param_1.device)) * theta_1.squeeze(
        -1
    )
    trace_1 = (w_1**2 + w_1**2) / 4

    mu_2, w_2, theta_2 = torch.split(param_2, (2, 1, 1), dim=-1)
    w_2 = w_2.squeeze(-1)
    theta_2 = torch.acos(torch.tensor(-1.0, device=param_2.device)) * theta_2.squeeze(
        -1
    )
    trace_2 = (w_2**2 + w_2**2) / 4
    sigma_1_sqrt = get_sigma_sqrt(w_1, theta_1)
    sigma_2 = get_sigma(w_2, theta_2)
    trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
    trace_12 = torch.sqrt(
        trace_12[..., 0, 0]
        + trace_12[..., 1, 1]
        + 2
        * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1]
            - trace_12[..., 0, 1] * trace_12[..., 1, 0]
        )
    )
    return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12


def pad(img: torch.Tensor, height: int, width: int):
    """Pad an image to a target height and width.

    Args:
        img (torch.Tensor): The image to pad.
        height (int): The target height.
        width (int): The target width.

    Returns:
        torch.Tensor: The padded image."""
    b, c, h, w = img.shape
    pad_h = (height - h) // 2
    pad_w = (width - w) // 2
    remainder_h = (height - h) % 2
    remainder_w = (width - w) % 2
    img = torch.cat(
        [
            torch.zeros((b, c, pad_h, w), device=img.device),
            img,
            torch.zeros((b, c, pad_h + remainder_h, w), device=img.device),
        ],
        dim=-2,
    )
    img = torch.cat(
        [
            torch.zeros((b, c, height, pad_w), device=img.device),
            img,
            torch.zeros((b, c, height, pad_w + remainder_w), device=img.device),
        ],
        dim=-1,
    )
    return img


def crop(img: torch.Tensor, height: int, width: int):
    H, W = img.shape[-2:]
    pad_h = (H - height) // 2
    pad_w = (W - width) // 2
    remainder_h = (H - height) % 2
    remainder_w = (W - width) % 2
    img = img[:, :, pad_h : H - pad_h - remainder_h, pad_w : W - pad_w - remainder_w]
    return img


def param2img_parallel(
    param: torch.Tensor,
    idx: torch.Tensor,
    decision: torch.Tensor,
    cur_canvas: torch.Tensor,
    environment: "Environment",
):
    """
    Input stroke parameters and decisions for each patch, meta brushes, current canvas, frame directory,
    and whether there is a border (if intermediate painting results are required).
    Output the painting results of adding the corresponding strokes on the current canvas.
    Args:
        param: a tensor with shape batch size x patch along height dimension x patch along width dimension
         x n_stroke_per_patch x n_param_per_stroke
        idx: Which shapes to render
        decision: a 01 tensor with shape batch size x patch along height dimension x patch along width dimension
         x n_stroke_per_patch
        cur_canvas: a tensor with shape batch size x 3 x H x W,
         where H and W denote height and width of padded results of original images.
        environment: The environment object containing the frame directory.


    Returns:
        cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
    """
    b, h, w, s, p = param.shape
    param = param.view(-1, 4).contiguous()
    idx = idx.view(-1, 1).contiguous()
    decision = decision.view(-1).contiguous().bool()
    H, W = cur_canvas.shape[-2:]
    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1
    patch_size_y = 2 * H // h
    patch_size_x = 2 * W // w
    even_idx_y = torch.arange(0, h, 2, device=cur_canvas.device)
    even_idx_x = torch.arange(0, w, 2, device=cur_canvas.device)
    odd_idx_y = torch.arange(1, h, 2, device=cur_canvas.device)
    odd_idx_x = torch.arange(1, w, 2, device=cur_canvas.device)
    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid(
        [even_idx_y, even_idx_x]
    )
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid([odd_idx_y, odd_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid([odd_idx_y, even_idx_x])
    cur_canvas = F.pad(
        cur_canvas,
        [
            patch_size_x // 4,
            patch_size_x // 4,
            patch_size_y // 4,
            patch_size_y // 4,
            0,
            0,
            0,
            0,
        ],
    )
    foregrounds = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x, device=cur_canvas.device
    )
    alphas = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x, device=cur_canvas.device
    )
    valid_foregrounds, valid_alphas = environment.param2stroke(
        param[decision, :],
        idx[decision, :],
        patch_size_y,
        patch_size_x,
        sample_theta=True,
    )
    foregrounds[decision, :, :, :] = valid_foregrounds
    alphas[decision, :, :, :] = valid_alphas
    foregrounds = foregrounds.view(
        -1, h, w, s, 3, patch_size_y, patch_size_x
    ).contiguous()
    alphas = alphas.view(-1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    decision = decision.view(-1, h, w, s, 1, 1, 1).contiguous()

    def partial_render(this_canvas, patch_coord_y, patch_coord_x):
        canvas_patch = F.unfold(
            this_canvas,
            (patch_size_y, patch_size_x),
            stride=(patch_size_y // 2, patch_size_x // 2),
        )
        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.view(
            b, 3, patch_size_y, patch_size_x, h, w
        ).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
        # canvas_patch: b, h, w, 3, py, px
        selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
        selected_foregrounds = foregrounds[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_alphas = alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_decisions = decision[:, patch_coord_y, patch_coord_x, :, :, :, :]
        for i in range(s):
            cur_foreground = selected_foregrounds[:, :, :, i, :, :, :]
            cur_alpha = selected_alphas[:, :, :, i, :, :, :]
            cur_decision = selected_decisions[:, :, :, i, :, :, :]
            selected_canvas_patch = (
                cur_foreground * cur_alpha * cur_decision
                + selected_canvas_patch * (1 - cur_alpha * cur_decision)
            )
        this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        # this_canvas: b, 3, h_half, py, w_half, px
        h_half = this_canvas.shape[2]
        w_half = this_canvas.shape[4]
        this_canvas = this_canvas.view(
            b, 3, h_half * patch_size_y, w_half * patch_size_x
        ).contiguous()
        # this_canvas: b, 3, h_half * py, w_half * px
        return this_canvas

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(
            cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x
        )
        if not is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2 :, : canvas.shape[3]]],
                dim=2,
            )
        if not is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, : canvas.shape[2], -patch_size_x // 2 :]],
                dim=3,
            )
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, : patch_size_y // 2, -canvas.shape[3] :], canvas], dim=2
        )
        canvas = torch.cat(
            [cur_canvas[:, :, -canvas.shape[2] :, : patch_size_x // 2], canvas], dim=3
        )
        if is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2 :, : canvas.shape[3]]],
                dim=2,
            )
        if is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, : canvas.shape[2], -patch_size_x // 2 :]],
                dim=3,
            )
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, : patch_size_y // 2, : canvas.shape[3]], canvas], dim=2
        )
        if is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2 :, : canvas.shape[3]]],
                dim=2,
            )
        if not is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, : canvas.shape[2], -patch_size_x // 2 :]],
                dim=3,
            )
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, : canvas.shape[2], : patch_size_x // 2], canvas], dim=3
        )
        if not is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2 :, -canvas.shape[3] :]],
                dim=2,
            )
        if is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, : canvas.shape[2], -patch_size_x // 2 :]],
                dim=3,
            )
        cur_canvas = canvas

    cur_canvas = cur_canvas[
        :,
        :,
        patch_size_y // 4 : -patch_size_y // 4,
        patch_size_x // 4 : -patch_size_x // 4,
    ]

    return cur_canvas


def debug_tensor_stats(name: str, tensor: torch.Tensor):
    """Print the min, max, and if the tensor is nan.

    Args:
        name (str): The name of the tensor.
        tensor (torch.Tensor): The tensor to print statistics for.
    """
    print(
        f"{name} min: {tensor.min().item()}, max: {tensor.max().item()}, nan: {torch.isnan(tensor).any().item()}"
    )


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


# Following classes are taken from
# https://github.com/toshas/torch_truncnorm
class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return (
            super(TruncatedNormal, self).log_prob(self._to_std_rv(value))
            - self._log_scale
        )
