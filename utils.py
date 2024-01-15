import argparse
import importlib
import omegaconf.dictconfig

from Register import Registers
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner
from tqdm import tqdm
import open3d as o3d
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import os
import torch
from pathlib import PosixPath
import numpy as np


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner


def _convert_depth_to_pc(depth_img, inverted_matrix):
    width, height, _ = depth_img.shape
    x_range = np.arange(width)
    y_range = np.arange(height)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    x_mesh_flat = x_mesh.reshape((-1,))
    y_mesh_flat = y_mesh.reshape((-1,))
    z_mesh = depth_img[y_mesh_flat, x_mesh_flat]

    x_mesh_flat = x_mesh_flat.reshape(-1, 1)
    y_mesh_flat = y_mesh_flat.reshape(-1, 1)
    z_mesh = z_mesh.reshape(-1, 1)
    np_ones = np.ones((x_mesh_flat.shape[0], 1))

    homo_points = np.hstack([x_mesh_flat, y_mesh_flat, z_mesh, np_ones])
    pc_points = (inverted_matrix @ homo_points.T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points[:, :3])
    return pcd


def convert_depth_to_pc(depth_path, min_bound_z=-0.2, outlier_neighbors=100, outlier_std_ratio=0.6, depth_res=512):
    matrix_2dto3d = np.asarray([[1 / depth_res, 0, 0, -0.5], [0, -1 / depth_res, 0, 0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])
    if isinstance(depth_path, str) or isinstance(depth_path, PosixPath):
        np_depth_map = np.load(str(depth_path))
    else:
        np_depth_map = depth_path
    wb_pcd = _convert_depth_to_pc(np_depth_map, matrix_2dto3d)
    np_vertices = np.asarray(wb_pcd.points)

    filtered_np_vertices = np_vertices[np_vertices[:, 2] > min_bound_z]
    wb_pcd.points = o3d.utility.Vector3dVector(filtered_np_vertices)
    cl, ind = wb_pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio)
    wb_filtered_pcd = wb_pcd.select_by_index(ind)

    wb_filtered_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    wb_filtered_pcd.estimate_normals()

    return wb_filtered_pcd

def save_pcd(pcd, dest_path):
    o3d.io.write_point_cloud(dest_path, pcd)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Convert depth to point cloud")
    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument("--id_wb", type=str, help="ID of woodblock")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_depth_path = os.path.join(args.data_dir, "200", f"{args.id_wb}.npy")
    gt_depth_path = os.path.join(args.data_dir, "ground_truth", f"{args.id_wb}.npy")
    gen_pcd = convert_depth_to_pc(gen_depth_path, outlier_std_ratio=0.5)
    gt_pcd = convert_depth_to_pc(gt_depth_path, outlier_std_ratio=5)
    o3d.visualization.draw_geometries([gt_pcd])
    o3d.visualization.draw_geometries([gen_pcd])
