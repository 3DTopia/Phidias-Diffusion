import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### LGM model setting
    # Unet input channel, image 3, ray 6, exclude ref maps
    unet_in_channel: int = 9
    # Unet image input size
    input_size: int = 320
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 160
    # gaussian render size
    output_size: int = 1024
    mixed_precision: Literal["fp32", "bf16"] = "bf16"

    ### Dataset settings
    # rendering root dir
    root_dirs: Tuple[str, ...] = (
        "path_to_rendered_images",
    )
    # data mode (only support s3 now)
    data_mode: Literal["s3"] = "s3"
    # fovy of the dataset
    fovy: float = 30  # 49.1 # 67.38013718106679
    # camera near plane 
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 10
    # number of views
    num_input_views: int = 6
    # camera radius
    cam_radius: float = 1.86603  # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### Inference setting
    # seed
    seed: int = 42
    # workspace, saved path
    workspace: str = "./workspace"
    # remove background
    rembg: bool = False
    # resize foreground ratio
    resize_fg_ratio: float = 0.85
    # diffusion steps
    diffusion_steps: int = 75
    # test image path
    test_path: Optional[str] = None

    ### Checkpoints
    # sparse view reconstruction model path
    sparse_view_recon_ckpt: Optional[str] = "model/lgm_6views_320x320.safetensors"
    # white-background multi-view unet path
    white_unet_path: Optional[str] = "model/zero123plus_finetuned_whitebg_unite_sphere.safetensors"
    # meta-controlnet path
    mv_controlnet_path:  Optional[str] = "model/meta_controlnet_v1.safetensors"

    ### Phidias model setting
    # multi-view model name
    mv_model_name: str = "zero123plus"
    # custom pipeline
    custom_pipeline: str = "phidias_pipeline"
    # use meta-control or not
    use_meta_control: bool = True
    # controlnet_conditioning_scale
    controlnet_conditioning_scale: float = 1.0
    # dynamic reference rounting
    downsample_size: Optional[Tuple[int, ...]] = (16,32,64)

    ### Retrieval setting
    # use retrieval
    use_retrieval: bool = False
    # top k retrieval
    top_k_retrieval: Optional[Tuple[int, ...]] = (1,)
    # objaverse point-clouds featture
    root_pcd_feats: Optional[str] = "model/objaverse-pcd-ours-feats-norgb"
    # retrieval precision
    retrieval_precision: str = 'fp16' # 'fp16' or 'fp32', 'fp36' would OOM on RTX4090 24G
    # rendering reference map online or not
    # if not, render all 3D reference under root_dir before testing
    online_rendering: bool = False
    # blender path
    blender_path: str = "blender-3.2.2-linux-x64/blender"
    # render azimuth
    render_azimuth: int = 0
    # render elevation
    render_elevation: int = 0
    
    ### others
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False


# all the default settings (for LGM)
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc["lrm"] = "the default settings for LGM"
config_defaults["lrm"] = Options()

config_doc["big"] = "big model with higher resolution Gaussians"
config_defaults["big"] = Options(
    input_size=320,
    up_channels=(1024, 1024, 512, 256, 128),  # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=160,
    output_size=1024,  # render & supervise Gaussians at a higher resolution.
    num_input_views=6,
    mixed_precision="bf16",
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
