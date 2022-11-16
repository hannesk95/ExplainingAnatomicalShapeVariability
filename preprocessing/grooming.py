import os
import argparse
import shapeworks as sw

from tqdm import tqdm
from glob import glob
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack


@contextmanager
def suppress_stdout(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def mkdir(path: str):
    """Checks whether directory already exists or not and creates it if not."""

    if not os.path.exists(path):
        os.makedirs(path)


def start_grooming(args):
    """Start data grooming pipeline."""
    mesh_dict = {}

    for segmentation in tqdm((glob(f"{args.data_dir}*.nii"))):
        with suppress_stdout():
            # Extract filename
            filename = segmentation.split("\\")[-1].split(".")[0].replace("_standard", "")

            # Read binary segmentation file
            shape_seg = sw.Image(segmentation)

            # Write file
            path = f"{args.output_dir}raw_meshes/"
            mkdir(path)
            mesh_shape = shape_seg.toMesh(1)
            mesh_shape.write(f"{path}{filename}.ply")

            # Binarize image
            shape_seg.binarize(0)

            # Padding
            shape_seg.pad(5, 0)

            # Compute distance transform
            shape_seg.computeDT(0)

            # Reduce high-frequency information
            shape_seg.gaussianBlur(0.05)

            # Convert into mesh
            mesh_shape = shape_seg.toMesh(1)

            # Write file
            path = f"{args.output_dir}after_image_level_preprocessing/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Smooth mesh surface
            mesh_shape.smooth(3, 1)

            # Write file
            path = f"{args.output_dir}after_surface_smoothing/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Center mesh
            center = mesh_shape.center()
            mesh_shape.translate(list(-center))

            # Write file
            path = f"{args.output_dir}after_centering/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Scale mesh
            bb = mesh_shape.boundingBox()
            bb.max.max() - bb.min.min()
            scale_factor = 1 / (bb.max.max() - bb.min.min())
            mesh_shape = mesh_shape.scale([scale_factor, scale_factor, scale_factor])

            # Write file
            path = f"{args.output_dir}after_scaling/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Append mesh to list
            mesh_dict[filename] = mesh_shape

    # Find reference medoid shape
    print("[INFO] Looking for reference (medoid) shape ...")
    mesh_list = list(mesh_dict.values())
    ref_index = sw.find_reference_mesh_index(mesh_list)
    ref_mesh = mesh_list[ref_index]
    ref_name = list(mesh_dict.keys())[ref_index]
    print(f"[INFO] Reference (medoid) shape found: {ref_name}")

    # Write file
    path = f"{args.output_dir}reference_medoid_shape/"
    mkdir(path)
    ref_mesh.write(f"{path}{ref_name}.ply")

    # Align all meshes to the reference medoid shape
    path = f"{args.output_dir}after_alignment/"
    mkdir(path)
    for name, mesh in tqdm(mesh_dict.items()):
        with suppress_stdout():
            # compute rigid transformation
            rigid_transform = mesh.createTransform(ref_mesh, sw.Mesh.AlignmentType.Rigid, 100)
            # apply rigid transform
            mesh.applyTransform(rigid_transform)
            mesh.write(f"{path}{name}.ply")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Grooming Pipeline')
    parser.add_argument('--data_dir', type=str, default='binary_segmentations/')
    parser.add_argument('--output_dir', type=str, default='groomed_data/')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start_grooming(args)
