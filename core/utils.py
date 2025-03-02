import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2._transform import Transform
from torchvision.transforms.v2._utils import query_chw
from torchvision.transforms.v2 import ColorJitter
from torch.utils._pytree import tree_flatten,tree_unflatten
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import roma
from kiui.op import safe_normalize
import rembg
import PIL.Image
from PIL import Image

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1).contiguous()  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, padding_mode='border', align_corners=False)

    return images


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

def remove_background(image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image


class MyRandomPhotometricDistort(Transform):
    """[BETA] Randomly distorts the image or video as used in `SSD: Single Shot
    MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

    .. v2betastatus:: RandomPhotometricDistort transform

    This transform relies on :class:`~torchvision.transforms.v2.ColorJitter`
    under the hood to adjust the contrast, saturation, hue, brightness, and also
    randomly permutes channels.

    Args:
        brightness (tuple of float (min, max), optional): How much to jitter brightness.
            brightness_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        contrast tuple of float (min, max), optional): How much to jitter contrast.
            contrast_factor is chosen uniformly from [min, max]. Should be non-negative numbers.
        saturation (tuple of float (min, max), optional): How much to jitter saturation.
            saturation_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        hue (tuple of float (min, max), optional): How much to jitter hue.
            hue_factor is chosen uniformly from [min, max].  Should have -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p (float, optional) probability each distortion operation (contrast, saturation, ...) to be applied.
            Default is 0.5.
    """

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.875, 1.125),
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        p: float = 0.5,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_channels, *_ = query_chw(flat_inputs)
        params: Dict[str, Any] = {
            key: ColorJitter._generate_value(range[0], range[1]) if torch.rand(1) < self.p else None
            for key, range in [
                ("brightness_factor", self.brightness),
                ("contrast_factor", self.contrast),
                ("saturation_factor", self.saturation),
                ("hue_factor", self.hue),
            ]
        }
        params["contrast_before"] = bool(torch.rand(()) < 0.5)
        params["channel_permutation"] = torch.randperm(num_channels) if torch.rand(1) < self.p else None
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        from torchvision.transforms.v2 import functional as F, Transform
        if params["brightness_factor"] is not None:
            inpt = self._call_kernel(F.adjust_brightness, inpt, brightness_factor=params["brightness_factor"])
        if params["contrast_factor"] is not None and params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["saturation_factor"] is not None:
            inpt = self._call_kernel(F.adjust_saturation, inpt, saturation_factor=params["saturation_factor"])
        if params["hue_factor"] is not None:
            inpt = self._call_kernel(F.adjust_hue, inpt, hue_factor=params["hue_factor"])
        if params["contrast_factor"] is not None and not params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["channel_permutation"] is not None:
            inpt = self._call_kernel(F.permute_channels, inpt, permutation=params["channel_permutation"])
        return inpt
    
    def forward(self, params=None, **kwargs: Any) -> Any:
        inputs=kwargs['inputs']
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        if params is None:
            params = self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec), params


from megfile import smart_open, smart_path_join, smart_exists
from megfile.errors import S3UnknownError
import time
def _locate_datadir(root_dirs, uid, locator: str):
    while True:
        try:
            for root_dir in root_dirs:
                datadir = smart_path_join(root_dir, uid, locator)
                if smart_exists(datadir):
                    return root_dir
            raise FileNotFoundError(
                f"Cannot find valid data directory for uid {uid}"
            )
        #  megfile.errors.S3UnknownError
        except S3UnknownError:
            print(f"S3UnknownError: {datadir}")
            time.sleep(5)

def _load_rgba_image(
    file_path,
    bg_color: float = 1.0,
    resize_fg_ratio=None,
    bg_remover=None
):
    """Load and blend RGBA image to RGB with certain background, 0-1 scaled"""
    img = Image.open(file_path)
    if resize_fg_ratio is not None:
        img = remove_background(img, bg_remover)
        img = resize_foreground(img, resize_fg_ratio)
    rgba = np.array(img)
    rgba = torch.from_numpy(rgba).float() / 255.0
    rgba = rgba.permute(2, 0, 1).contiguous()
    mask = rgba[3:4]  # [1, 512, 512]
    rgb = rgba[:3, :, :]

    rgb = rgb * mask + bg_color * (1 - mask)
    return rgb, mask

import subprocess

def run_bash(bash_args, logger=None):
    process = subprocess.Popen(
        bash_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )

    while True:
        if process.poll() is not None:
            break

        output = process.stdout.readline().decode("utf-8")
        print(output, end="")
        if logger is not None:
            logger.write(output)
            logger.flush()