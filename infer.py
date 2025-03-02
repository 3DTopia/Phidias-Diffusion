import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import kiui
from core.utils import spherical_camera_pose, _load_rgba_image, _locate_datadir, run_bash
from core.options import AllConfigs, Options
from core.models import LGM

from PIL import Image
import pillow_avif
from einops import rearrange, repeat
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from core.utils import remove_background, resize_foreground
from torchvision.transforms import v2
from safetensors.torch import load_file
import objaverse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# load configs
opt = tyro.cli(AllConfigs)

# set seed & device
kiui.utils.seed_everything(opt.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Sparse-View 3D Reconstruction Model
sparse_view_recon_model = LGM(opt)
if opt.sparse_view_recon_ckpt is not None:
    ckpt = load_file(opt.sparse_view_recon_ckpt, device="cpu")
    sparse_view_recon_model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint from {opt.sparse_view_recon_ckpt}")
else:
    print(f"[WARN] sparse_view_recon_model randomly initialized, are you sure?")

sparse_view_recon_model = sparse_view_recon_model.half().to(device)
sparse_view_recon_model.eval()


# Load Multi-View Diffusion Model
print("Loading multi-view diffusion model ...")
if opt.mv_model_name == "zero123plus":
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline=opt.custom_pipeline,
        torch_dtype=torch.float16,
    )

    # load custom white-background unet
    if opt.white_unet_path is not None:
        print(f"Loading custom white-background unet from {opt.white_unet_path}...")
        state_dict = load_file(opt.white_unet_path)
        pipeline.unet.load_state_dict(state_dict, strict=True)

    # load meta-ControlNet
    if opt.mv_controlnet_path is not None:
        pipeline.add_controlnet(
            conditioning_scale=opt.controlnet_conditioning_scale,
            use_meta_control=opt.use_meta_control,
        )
        pipeline.unet.controlnet.to(torch.float16)

        assert os.path.exists(opt.mv_controlnet_path)
        print(f"Loading custom mv controlnet from {opt.mv_controlnet_path}...")

        state_dict = load_file(opt.mv_controlnet_path)
        pipeline.unet.controlnet.load_state_dict(state_dict, strict=True)

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )

    pipe = pipeline
else:
    raise ValueError(f"Unsupported mv_model_name: {opt.mv_model_name}")

pipe = pipe.to(device)

# load backgrond remover
bg_remover = None if not opt.rembg else rembg.new_session()


# process function
def process(opt: Options, path, top_k_retrieval=None, retrieval_model=None):
    name = os.path.splitext(os.path.basename(path))[0]

    # skip generated
    if os.path.exists(os.path.join(opt.workspace, name + ".mp4" if not opt.use_retrieval else name + f"_top{top_k_retrieval}" + ".mp4")):
        print(f"[INFO] Skip processing {path} --> {name}")
        return

    print(f"[INFO] Processing {path} --> {name}")
    os.makedirs(opt.workspace, exist_ok=True)

    ################## Multi-view Image Prediction ##################
    if opt.mv_model_name == "zero123plus":
        input_image = Image.open(path)
        if opt.rembg:
            print("remove background")
            input_image = remove_background(input_image, bg_remover)
            input_image = resize_foreground(input_image, opt.resize_fg_ratio)

        # 1. use 3D reference
        if opt.mv_controlnet_path is not None:
            ref_obj_name = name

            if name.split("_")[-1].isdigit():  # 3D-to-3D file name processor
                ref_obj_name = "_".join(name.split("_")[:-1])

            # use user-specific 3D reference (e.g., 3D-to-3D)
            if not opt.use_retrieval:
                ref_map_path_base = os.path.join(
                    os.path.dirname(path),
                    "../ref_maps_user",
                    ref_obj_name,
                )
            else:  # use retrieved 3D reference from objaverse subset
                # conduct retrieval
                ref_uid = retrieval_model.retrieve_image_to_pcd(path, n_candidates=top_k_retrieval)[
                    -1
                ]
                # read rendered reference maps
                if not opt.online_rendering:  # load from pre-rendered path
                    ref_root_dir = _locate_datadir(
                        opt.root_dirs, ref_uid, locator="intrinsics.npy"
                    )
                else:
                    print(f"Rendering reference maps for {ref_uid} ...")
                    obj_path = objaverse.load_objects([ref_uid])[ref_uid]
                    print(f'Object saved at {obj_path}')
                    print(f'Start rendering ...')
                    render_args = [
                        f"{opt.blender_path} -b -P scripts/blender_script.py -- --object_path {obj_path} "
                        f"--output_dir data_test/ref_maps_retrieved "
                        f"--resolution 512 "
                        f"--num_images 8 "
                        f"--start_azimuth {opt.render_azimuth} "
                        f"--start_elevation {opt.render_elevation}"
                    ]
                    print(render_args)
                    run_bash(render_args)
                    ref_root_dir = os.path.join("data_test", "ref_maps_retrieved")
                    print(f'Rendering finished. Reference maps saved at data_test/ref_maps/{ref_uid}')

                ref_map_path_base = os.path.join(ref_root_dir, ref_uid)

            imgs = []
            for i in [0, 2, 3, 4, 5, 6, 7]:  # 1 front view + 6 side views
                ref_map_path = os.path.join(
                    ref_map_path_base, "ccm", f"{i:03d}_0001.png"
                )
                imgs.append(
                    _load_rgba_image(
                        ref_map_path,
                        bg_color=1.0,
                    )[0]
                )
            imgs = torch.stack(imgs, dim=0).float()
            refs = torch.cat([x for x in [imgs]], dim=1)  # (7, 3*num_ref_maps, H, W)

            ref_img_dict = dict()
            meta_controller_input_dict = dict()
            # Dynamic-Reference Rounting
            for downsample_size in opt.downsample_size:
                # downsample to 16x16, 32x32, 64x64
                downsampled_ref = F.interpolate(
                    refs,
                    size=(
                        downsample_size,
                        downsample_size,
                    ),
                    mode="nearest",
                )  # [V, C, H, W]
                # upsample to 320
                upsampled_ref = F.interpolate(
                    downsampled_ref,
                    size=(
                        320,
                        320,
                    ),
                    mode="nearest",
                )  # [V, C, H, W]
                all_ref_imgs = upsampled_ref.clamp(0, 1)
                ref_imgs = all_ref_imgs[1:]
                ref_imgs = rearrange(
                    ref_imgs, "(x y) c h w -> c (x h) (y w)", x=3, y=2
                )  # (C, 3H, 2W)
                ref_imgs = [ref_imgs[i : i + 3] for i in range(0, ref_imgs.shape[0], 3)]
                ref_imgs_vis = ref_imgs
                ref_imgs = [v2.functional.to_pil_image(i) for i in ref_imgs]
                ref_img_dict[f"ref_imgs_{downsample_size}"] = ref_imgs

                # read meta-controller inputs (i.e., ref_cond_align_imgs)
                # ref_cond_align_imgs = [concept image (i.e., cond_front_imgs), front-view ccm (i.e., ref_front_imgs)]
                if opt.use_meta_control:
                    # ref_front_imgs
                    ref_front_imgs = all_ref_imgs[0:1].repeat(
                        6, 1, 1, 1
                    )  # (6, 3*num_ref_maps, H, W)
                    ref_front_imgs = rearrange(
                        ref_front_imgs, "(x y) c h w -> c (x h) (y w)", x=3, y=2
                    )  # (C, 3H, 2W)

                    ref_front_imgs = [
                        ref_front_imgs[i : i + 3]
                        for i in range(0, ref_front_imgs.shape[0], 3)
                    ]
                    ref_cond_align_imgs_vis = [i[:, :320, :320] for i in ref_front_imgs]
                    ref_front_imgs = [
                        v2.functional.to_pil_image(i) for i in ref_front_imgs
                    ]

                    cond_front_imgs = _load_rgba_image(
                        path,
                        bg_color=1.0,
                        resize_fg_ratio=(None if not opt.rembg else opt.resize_fg_ratio),
                        bg_remover=bg_remover,
                    )[0].unsqueeze(
                        0
                    )  # (1, C, H, W)
                    cond_front_imgs = (
                        v2.functional.resize(
                            cond_front_imgs, 320, interpolation=3, antialias=True
                        )
                        .clamp(0, 1)
                        .repeat(6, 1, 1, 1)
                    )
                    cond_front_imgs = rearrange(
                        cond_front_imgs, "(x y) c h w -> c (x h) (y w)", x=3, y=2
                    )  # (C, 3H, 2W)

                    cond_front_imgs_vis = cond_front_imgs.clone()[:, :320, :320]
                    ref_cond_align_imgs_vis = [cond_front_imgs_vis] + ref_cond_align_imgs_vis
                    ref_cond_align_imgs_vis = torch.cat(ref_cond_align_imgs_vis, dim=2)
                    cond_front_imgs = v2.functional.to_pil_image(cond_front_imgs)

                # visualization
                if downsample_size == opt.downsample_size[-1]:
                    if opt.use_retrieval:
                        save_ref_path = os.path.join(
                            opt.workspace,
                            name
                            + f"_top{top_k_retrieval}"
                            + f"_mv_refs_{downsample_size}+_{ref_uid}"
                            + ".png",
                        )
                    else:
                        save_ref_path = os.path.join(
                            opt.workspace,
                            name + f"_mv_refs_{downsample_size}" + ".png",
                        )

                    if opt.use_meta_control:
                        ref_imgs_vis = [ref_cond_align_imgs_vis] + ref_imgs_vis

                    v2.functional.to_pil_image(torch.cat(ref_imgs_vis, dim=1)).save(
                        save_ref_path
                    )
                    print(
                        f"{opt.custom_pipeline} mv reference image saved to {save_ref_path}"
                    )

                if opt.use_meta_control:
                    meta_controller_input_dict[
                        f"ref_cond_align_imgs_{downsample_size}"
                    ] = {
                        "conds_front_img": cond_front_imgs,
                        "ref_front_img": ref_front_imgs,
                    }
                else:
                    meta_controller_input_dict = None

            mv_image = pipe(
                input_image,
                depth_image_dict=ref_img_dict,
                meta_controller_input_dict=meta_controller_input_dict,
                num_inference_steps=opt.diffusion_steps,
            ).images[0]

        # 2. without 3D reference
        else:
            mv_image = pipe(
                input_image,
                prompt="",
                num_inference_steps=opt.diffusion_steps,
            ).images[0]

        mv_image = np.asarray(mv_image, dtype=np.float32) / 255.0
        mv_image = rearrange(
            mv_image, "(n h) (m w) c -> (n m) h w c", n=3, m=2
        )  # (num_views, 256, 256, 3)
    else:
        raise ValueError(f"Unsupported mv_model_name: {opt.mv_model_name}")

    kiui.write_image(
        os.path.join(
            opt.workspace,
            (
                name + ".png"
                if not opt.use_retrieval
                else name + f"_top{top_k_retrieval}" + ".png"
            ),
        ),
        mv_image.reshape(-1, mv_image.shape[1], 3),
    )
    print(f"MV image saved to {os.path.join(opt.workspace, name + '.png')}")

    ################## Sparse-View 3D Reconstruction ##################
    rays_embeddings = sparse_view_recon_model.prepare_default_rays(device)

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    input_image = (
        torch.from_numpy(mv_image).permute(0, 3, 1, 2).contiguous().float().to(device)
    )  # [4/6, 3, 256, 256]
    input_image = F.interpolate(
        input_image,
        size=(opt.input_size, opt.input_size),
        mode="bilinear",
        align_corners=False,
    )
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(
        0
    )  # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # generate gaussians
            gaussians = sparse_view_recon_model.forward_gaussians(input_image)

        # save gaussians
        sparse_view_recon_model.gs.save_ply(
            gaussians, os.path.join(opt.workspace, name + ".ply")
        )

        # render 360 video
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):

                cam_poses = (
                    spherical_camera_pose(azi, elevation, radius=opt.cam_radius)
                    .unsqueeze(0)
                    .to(device)
                )

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = (
                    torch.inverse(cam_poses).transpose(1, 2).contiguous()
                )  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

                scale = min(azi / 360, 1)

                image = sparse_view_recon_model.gs.render(
                    gaussians,
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                    scale_modifier=scale,
                )["image"]
                images.append(
                    (
                        image.squeeze(1)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .float()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):

                cam_poses = (
                    spherical_camera_pose(azi, elevation, radius=opt.cam_radius)
                    .unsqueeze(0)
                    .to(device)
                )

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = (
                    torch.inverse(cam_poses).transpose(1, 2).contiguous()
                )  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

                image = sparse_view_recon_model.gs.render(
                    gaussians,
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                    scale_modifier=1,
                )["image"]
                images.append(
                    (
                        image.squeeze(1)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .float()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )

        images = np.concatenate(images, axis=0)
        if opt.use_retrieval:
            imageio.mimwrite(
                os.path.join(opt.workspace, name + f"_top{top_k_retrieval}" + ".mp4"),
                images,
                fps=30,
            )
        else:
            imageio.mimwrite(os.path.join(opt.workspace, name + ".mp4"), images, fps=30)


assert opt.test_path is not None

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]

# load retrieval model
if opt.use_retrieval:
        print('Loading retrieval model ... This may take a while.')
        from scripts.retrieve_ref_from_img import RetrievalImageToPointCloud
        retrieval_model = RetrievalImageToPointCloud(device="cuda", root_pcd_feats=opt.root_pcd_feats, precision=opt.retrieval_precision)

for path in file_paths:
    if opt.use_retrieval:
        for top_k_retrieval in opt.top_k_retrieval:
            process(opt, path, top_k_retrieval=top_k_retrieval, retrieval_model=retrieval_model)
    else:
        process(opt, path)
