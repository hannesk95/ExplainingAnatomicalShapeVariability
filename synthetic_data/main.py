import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import pyvista as pv

from sdf.sdf import *
from glob import glob
from tqdm import tqdm
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='Synthetic Dataset')
parser.add_argument('--dataset', type=str, default='all', choices=['all', 'box_bump', 'torus_bump'])
args = parser.parse_args()


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def create_box_bump_files():
    """Create synthetic box data."""

    if not os.path.exists("box_bump"):
        os.makedirs("box_bump")

    # Create template file
    with suppress_stdout():
        filepath = "box_bump/box_bump_template.ply"
        a = rounded_box((1, 0.5, 0.5), 0.1)
        a.save(filepath, samples=30000)

    # Create mesh files and save labels
    labels = {}
    feature_range = np.linspace(-0.3, 0.3, 500)
    scaler = MinMaxScaler()
    scaler.fit(feature_range.reshape(-1, 1))

    i = 0
    for x in tqdm(feature_range):
        with suppress_stdout():
            filepath = f'box_bump/box_bump_{i:03d}.ply'
            filename = filepath.split("/")[-1].split(".")[0]
            b = sphere(0.075).translate((x, 0.275, 0))
            f = union(a, b, k=0.15)
            f.save(filepath, samples=20000)
            labels[filename] = scaler.transform(np.array([x]).reshape(-1, 1)).item()
            i += 1

    torch.save(labels, "box_bump/box_bump_labels.pt")
    pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv("box_bump/box_bump_labels.csv")

    # Convert meshes into .vtk file format
    if not os.path.exists("box_bump\\vtk_files"):
        os.makedirs("box_bump\\vtk_files")

    for mesh_path in tqdm(glob("box_bump\\*.ply")):
        filename = mesh_path.split("\\")[-1].split(".")[0]
        mesh = pv.read(mesh_path)
        mesh.save(f"box_bump\\vtk_files\\{filename}.vtk")


def create_torus_bump_files():
    """Create synthetic torus data."""

    if not os.path.exists("torus_bump"):
        os.makedirs("torus_bump")

    # Create template file
    with suppress_stdout():
        filepath = "torus_bump/torus_bump_template.ply"
        a = torus(1, 0.35)
        a.save(filepath, samples=30000)

    # Create mesh files and save labels
    labels = {}
    feature_range = np.linspace(0, 360, 500, endpoint=False)
    scaler = MinMaxScaler()
    scaler.fit(feature_range.reshape(-1, 1))

    i = 0
    for angle in tqdm(feature_range):
        with suppress_stdout():
            filepath = f'torus_bump/torus_bump_{i:03d}.ply'
            filename = filepath.split("/")[-1].split(".")[0]
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            labels[filename] = scaler.transform(np.array([angle]).reshape(-1, 1)).item()

            b = sphere(0.6).translate((x, y, 0))
            f = union(a, b, k=0.2)
            f.save(filepath, samples=20000)
            i += 1

    torch.save(labels, "torus_bump\\torus_bump_labels.pt")
    pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv("torus_bump\\torus_bump_labels.csv")

    # Convert meshes into .vtk file format
    if not os.path.exists("torus_bump\\vtk_files"):
        os.makedirs("torus_bump\\vtk_files")

    for mesh_path in tqdm(glob("torus_bump\\*.ply")):
        filename = mesh_path.split("\\")[-1].split(".")[0]
        mesh = pv.read(mesh_path)
        mesh.save(f"torus_bump\\vtk_files\\{filename}.vtk")


if __name__ == "__main__":

    if args.dataset == "all":
        create_box_bump_files()
        create_torus_bump_files()

    elif args.dataset == "box_bump":
        create_box_bump_files()

    elif args.dataset == "torus_bump":
        create_torus_bump_files()
