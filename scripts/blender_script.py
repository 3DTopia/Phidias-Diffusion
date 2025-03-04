"""
Blender script to render images of 3D models.

Modified from https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py

Example usage:
    blender -b -P blender_script.py -- \
            --object_path xxx.glb \
            --output_dir output_dir \
            --resolution 512 \
            --num_images 8  \
            --start_azimuth 0 \
            --start_elevation 0 \

"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import bmesh

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--fov", type=int, default=30)
parser.add_argument("--start_azimuth", type=float, default=0)
parser.add_argument("--start_elevation", type=float, default=0)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print("===================", args.engine, "===================")

context = bpy.context
scene = context.scene
render = scene.render
# for normal, depth, ccm rendering
scene.use_nodes = True
context.view_layer.use_pass_normal = True  # for normal rendering
context.view_layer.use_pass_z = False  # for depth rendering
context.view_layer.use_pass_position = True  # for ccm rendering
scene.view_settings.view_transform = "Standard"  # for rgba

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))


def sample_point_on_sphere(
    radius: float, theta: float = None, phi: float = None
) -> Tuple[float, float, float]:
    if theta is None:
        theta = random.random() * 2 * math.pi
    if phi is None:
        phi = math.acos(2 * random.random() - 1)
    # print(theta, phi)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)

        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def set_camera_location(camera, elevation, azimuth, radius):

    ele = np.deg2rad(elevation)
    azi = np.deg2rad(azimuth)
    x, y, z = sample_point_on_sphere(
        radius, theta=np.deg2rad(-90) + azi, phi=np.deg2rad(90) - ele
    )

    camera.location = x, y, z

    # adjust orientation
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera, (x, y, z)


def set_camera_location_xyz(camera, _xyz):
    x, y, z = _xyz[0], _xyz[1], _xyz[2]

    camera.location = x, y, z

    # adjust orientation
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    use_contact_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.use_contact_shadow = use_contact_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    if light_type == "SUN":
        light_data.angle = 1.57
    return light_object


def randomize_lighting():
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    # Add random angle offset in 0-90
    angle_offset = random.uniform(0, math.pi / 2)

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398 + angle_offset),  # 45 0 -45
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699 + angle_offset),  # -45 0 -225
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619 + angle_offset),  # 45 0 135
        energy=random.choice([2, 3, 3.5]),
    )

    # Create small light
    small_light1 = _create_light(
        name="S1_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 0.785398 + angle_offset),  # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    small_light2 = _create_light(
        name="S2_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 3.92699 + angle_offset),  # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),  # 180 0 0
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
        small_light1=small_light1,
        small_light2=small_light2,
    )


def add_lighting(option: str) -> None:
    assert option in ["fixed", "random"]

    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == "fixed":
        light.energy = 30000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 1
        bpy.data.objects["Area"].location[2] = 0.5

    elif option == "random":
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2.0, 2.0)
        bpy.data.objects["Area"].location[1] = random.uniform(-2.0, 2.0)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    # set light scale
    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path, forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    max_dist = -math.inf
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
            max_dist = max(max_dist, (coord - Vector((0.0, 0.0, 0.0))).length_squared)
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max), np.sqrt(max_dist)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene_box(box_scale: float):
    bbox_min, bbox_max, _ = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max, _ = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def normalize_scene_sphere(radius: float):
    normalize_scene_box(1)

    bbox_min, bbox_max, max_dist = scene_bbox()
    center = (bbox_min + bbox_max) / 2
    scale = radius / max_dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max, _ = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)

    # cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = (
        32  # affects instrinsics calculation, should be set explicitly
    )

    # Convert FOV to radians
    assert args.fov == 30
    fov_radians = np.deg2rad(args.fov)

    # set FOV
    cam.data.angle = fov_radians

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(object_file: str, extra_outs=["normal", "ccm"]) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    if object_uid == "model":
        object_uid = object_file.split("/")[-3]
    object_uid = "_".join(object_uid.lower().split())
    unit_sphere = True
    if unit_sphere:
        normalize_scene_sphere(radius=0.5)
    else:
        normalize_scene_box(box_scale=2)

    # add_lighting(option="random")
    randomize_lighting()
    camera, cam_constraint = setup_camera()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # prepare to save
    img_dir = os.path.join(args.output_dir, object_uid, "rgba")
    pose_dir = os.path.join(args.output_dir, object_uid, "pose")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    scene.use_nodes = False
    context.view_layer.use_pass_normal = False  # for normal rendering
    context.view_layer.use_pass_position = False  # for ccm rendering
    camera_xyz = []
    time_0 = time.time()
    for i in range(args.num_images):
        # set the camera position
        fixed_elevations = np.array(
            [0.0, 0, 20, -10, 20, -10, 20, -10, -10, 20, -10, 20, -10, 20]
        )
        fixed_azimuths = np.array(
            [0.0, 180, 30, 90, 150, 210, 270, 330, 30, 90, 150, 210, 270, 330]
        )

        if i < len(fixed_elevations):
            elevation = fixed_elevations[i]
            azimuth = fixed_azimuths[i]
        else:
            while True:
                elevation = np.random.uniform(-20, 50)
                azimuth = np.random.uniform(0, 360)
                if np.any((np.abs(fixed_azimuths - int(azimuth))) < 5) and np.any(
                    (np.abs(fixed_elevations - int(elevation))) < 5
                ):
                    continue
                else:
                    break
        radius = 0.5 / np.tan(np.radians(args.fov / 2))
        camera, _xyz = set_camera_location(
            camera,
            elevation + args.start_elevation,
            azimuth + args.start_azimuth,
            radius,
        )
        camera_xyz.append(_xyz)

        # render the image
        render_path = os.path.join(img_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix (C2W)
        location, rotation = camera.matrix_world.decompose()[0:2]
        RT = compose_RT(rotation.to_matrix(), np.array(location))
        RT_path = os.path.join(pose_dir, f"{i:03d}_mat.npy")
        np.save(RT_path, RT)

        # raw camera pose: elevation, azimuth, radius
        pose = [elevation, azimuth, radius]
        pose_path = os.path.join(pose_dir, f"{i:03d}_raw.npy")
        np.save(pose_path, pose)

    time_0_ = time.time() - time_0

    scene.use_nodes = True
    context.view_layer.use_pass_normal = True  # for normal rendering
    context.view_layer.use_pass_position = True  # for ccm rendering

    extra_dirs = {}
    for t in extra_outs:
        temp_dir = os.path.join(args.output_dir, object_uid, t)
        os.makedirs(temp_dir, exist_ok=True)
        extra_dirs[t] = temp_dir

    # create input render layer node
    render_layers = scene.node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.label = "Custom Outputs"
    render_layers.name = "Custom Outputs"

    if "normal" in extra_outs:
        # create normal output node
        normal_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "normal"
        normal_file_output.name = "normal"
        normal_file_output.base_path = ""

        # Create a Separate RGB node
        separate_rgb_node = scene.node_tree.nodes.new(type="CompositorNodeSepRGBA")
        scene.node_tree.links.new(
            render_layers.outputs["Normal"], separate_rgb_node.inputs["Image"]
        )

        # Create a Combine RGBA node
        combine_rgba_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
        reversed_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        reversed_G.operation = "MULTIPLY"
        reversed_G.inputs[0].default_value = -1

        # y -> -y
        scene.node_tree.links.new(separate_rgb_node.outputs["G"], reversed_G.inputs[1])

        # map normal range from [-1, 1] to [0, 1]
        # channel R
        bias_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_R.operation = "ADD"
        bias_node_R.inputs[0].default_value = 1

        scale_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_R.operation = "MULTIPLY"
        scale_node_R.inputs[0].default_value = 0.5

        scene.node_tree.links.new(separate_rgb_node.outputs["R"], bias_node_R.inputs[1])
        scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

        # channel G
        bias_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_G.operation = "ADD"
        bias_node_G.inputs[0].default_value = 1

        scale_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_G.operation = "MULTIPLY"
        scale_node_G.inputs[0].default_value = 0.5

        scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
        scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

        # channel B
        bias_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_B.operation = "ADD"
        bias_node_B.inputs[0].default_value = 1

        scale_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_B.operation = "MULTIPLY"
        scale_node_B.inputs[0].default_value = 0.5

        scene.node_tree.links.new(separate_rgb_node.outputs["B"], bias_node_B.inputs[1])
        scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])

        # Combine RGB
        scene.node_tree.links.new(
            combine_rgba_node.inputs["R"], scale_node_R.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["G"], scale_node_B.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["B"], scale_node_G.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["A"], separate_rgb_node.outputs["A"]
        )

        # add alpha channel
        set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_alpha_node.mode = "REPLACE_ALPHA"
        scene.node_tree.links.new(
            combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
        )

        scene.node_tree.links.new(
            set_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
        )

    if "ccm" in extra_outs:
        # create CCM output node
        ccm_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        ccm_file_output.label = "ccm"
        ccm_file_output.name = "ccm"
        ccm_file_output.base_path = ""

        # Create a Separate RGB node
        separate_xyz_node = scene.node_tree.nodes.new(type="CompositorNodeSeparateXYZ")
        scene.node_tree.links.new(
            render_layers.outputs["Position"], separate_xyz_node.inputs["Vector"]
        )

        # Create a Combine RGBA node
        combine_rgba_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
        reversed_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        reversed_G.operation = "MULTIPLY"
        reversed_G.inputs[0].default_value = -1

        # y -> -y
        scene.node_tree.links.new(separate_xyz_node.outputs["Y"], reversed_G.inputs[1])

        # map ccm range from [-1, 1] to [0, 1] (normalize scence to unit box)
        # map ccm range from [-0.5, 0.5] to [0, 1] (normalize scence to unit sphere)
        if unit_sphere:
            add = 0.5
            mul = 1
        else:
            add = 1
            mul = 0.5

        # channel R
        bias_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_R.operation = "ADD"
        bias_node_R.inputs[0].default_value = add

        scale_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_R.operation = "MULTIPLY"
        scale_node_R.inputs[0].default_value = mul

        scene.node_tree.links.new(separate_xyz_node.outputs["X"], bias_node_R.inputs[1])
        scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

        # channel G
        bias_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_G.operation = "ADD"
        bias_node_G.inputs[0].default_value = add

        scale_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_G.operation = "MULTIPLY"
        scale_node_G.inputs[0].default_value = mul

        scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
        scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

        # channel B
        bias_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        bias_node_B.operation = "ADD"
        bias_node_B.inputs[0].default_value = add

        scale_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
        scale_node_B.operation = "MULTIPLY"
        scale_node_B.inputs[0].default_value = mul

        scene.node_tree.links.new(separate_xyz_node.outputs["Z"], bias_node_B.inputs[1])
        scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])

        # Combine RGB
        scene.node_tree.links.new(
            combine_rgba_node.inputs["R"], scale_node_R.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["G"], scale_node_B.outputs[0]
        )
        scene.node_tree.links.new(
            combine_rgba_node.inputs["B"], scale_node_G.outputs[0]
        )

        # add alpha channel
        set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_alpha_node.mode = "REPLACE_ALPHA"
        scene.node_tree.links.new(
            combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
        )
        scene.node_tree.links.new(
            render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
        )

        scene.node_tree.links.new(
            set_alpha_node.outputs["Image"], ccm_file_output.inputs["Image"]
        )

    time_2 = time.time()
    if len(extra_outs) > 0:
        scene.view_settings.view_transform = "Raw"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        for i, _xyz in enumerate(camera_xyz):
            # set the camera position
            camera = set_camera_location_xyz(camera, _xyz)

            # render the image
            scene.render.filepath = ".cache/tmp.png"

            for out in extra_outs:
                render_path = os.path.join(extra_dirs[out], f"{i:03d}_")
                scene.node_tree.nodes[out].file_slots[0].path = render_path

            bpy.ops.render.render(write_still=True)
        scene.view_settings.view_transform = "Standard"
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
    time_2_ = time.time() - time_2

    # print("0: Time taken for rendering", time_0_, "seconds")
    # print("2: Time taken for rendering", time_2_, "seconds")
    # save the camera intrinsics
    intrinsics = get_calibration_matrix_K_from_blender(
        camera.data, return_principles=True
    )
    with open(
        os.path.join(args.output_dir, object_uid, "intrinsics.npy"), "wb"
    ) as f_intrinsics:
        np.save(f_intrinsics, intrinsics)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    """
    Get the camera intrinsic matrix from Blender camera.
    Return also numpy array of principle parameters if specified.

    Intrinsic matrix K has the following structure in pixels:
        [fx  0 cx]
        [0  fy cy]
        [0   0  1]

    Specified principle parameters are:
        [fx, fy] - focal lengths in pixels
        [cx, cy] - optical centers in pixels
        [width, height] - image resolution in pixels

    """
    # Render resolution
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y

    # Camera parameters
    focal_length = camera.lens  # Focal length in millimeters
    sensor_width = camera.sensor_width  # Sensor width in millimeters
    sensor_height = camera.sensor_height  # Sensor height in millimeters

    # Calculate the focal length in pixel units
    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)

    # Assuming the optical center is at the center of the sensor
    optical_center_x = width / 2
    optical_center_y = height / 2

    # Constructing the intrinsic matrix
    K = np.array(
        [
            [focal_length_x, 0, optical_center_x],
            [0, focal_length_y, optical_center_y],
            [0, 0, 1],
        ]
    )

    if return_principles:
        return np.array(
            [
                [focal_length_x, focal_length_y],
                [optical_center_x, optical_center_y],
                [width, height],
            ]
        )
    else:
        return K


if __name__ == "__main__":
    start_i = time.time()
    if args.object_path.startswith("http"):
        local_path = download_object(args.object_path)
    else:
        local_path = args.object_path
    save_images(local_path, extra_outs=["ccm"])
    end_i = time.time()
    # print("Finished", local_path, "in", end_i - start_i, "seconds")
    # delete the object if it was downloaded
    if args.object_path.startswith("http"):
        os.remove(local_path)
