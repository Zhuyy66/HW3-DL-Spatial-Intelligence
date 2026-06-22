"""
Run inside Blender:

blender --python blender_import_cameras.py -- \
  --transforms_json /abs/path/transforms_blender.json \
  --camera_name CounterCamera \
  --collection_name CounterTrajectory
"""

import argparse
import json
import math
import sys

import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Import animated camera trajectory into Blender.")
    parser.add_argument("--transforms_json", required=True)
    parser.add_argument("--camera_name", default="ImportedCamera")
    parser.add_argument("--collection_name", default="ImportedTrajectory")
    parser.add_argument("--sensor_width", type=float, default=36.0)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--create_markers", action="store_true")
    return parser.parse_args(argv)


def ensure_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def clear_object_if_exists(name):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def make_camera(camera_name, sensor_width):
    clear_object_if_exists(camera_name)
    camera_data = bpy.data.cameras.new(camera_name)
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = sensor_width
    camera_obj = bpy.data.objects.new(camera_name, camera_data)
    return camera_obj


def make_trajectory_curve(points, name):
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for idx, p in enumerate(points):
        spline.points[idx].co = (p[0], p[1], p[2], 1.0)
    curve_obj = bpy.data.objects.new(name, curve_data)
    return curve_obj


def add_marker(collection, name, location, frame):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.05
    empty.location = location
    empty["frame"] = frame
    collection.objects.link(empty)


def apply_frame(camera_obj, frame_idx, frame, sensor_width):
    matrix = Matrix(frame["transform_matrix"])
    camera_obj.matrix_world = matrix

    width = frame["width"]
    fx = frame["fx"]
    lens_mm = fx * sensor_width / width
    camera_obj.data.lens = lens_mm
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = 1000.0

    camera_obj.keyframe_insert(data_path="location", frame=frame_idx)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    camera_obj.data.keyframe_insert(data_path="lens", frame=frame_idx)


def main():
    args = parse_args()
    with open(args.transforms_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"]
    if not frames:
        raise ValueError("No frames found in transforms JSON.")

    collection = ensure_collection(args.collection_name)
    camera_obj = make_camera(args.camera_name, args.sensor_width)
    collection.objects.link(camera_obj)

    points = []
    for offset, frame in enumerate(frames):
        frame_idx = args.frame_start + offset
        apply_frame(camera_obj, frame_idx, frame, args.sensor_width)
        loc = Vector(frame["position"])
        points.append(loc)
        if args.create_markers:
            add_marker(collection, f"cam_{offset:04d}", loc, frame_idx)

    curve_obj = make_trajectory_curve(points, f"{args.camera_name}_trajectory")
    collection.objects.link(curve_obj)

    bpy.context.scene.camera = camera_obj
    bpy.context.scene.frame_start = args.frame_start
    bpy.context.scene.frame_end = args.frame_start + len(frames) - 1
    bpy.context.scene.render.resolution_x = frames[0]["width"]
    bpy.context.scene.render.resolution_y = frames[0]["height"]

    print(
        f"Imported {len(frames)} frames from {args.transforms_json} "
        f"into collection {args.collection_name}"
    )


if __name__ == "__main__":
    main()
