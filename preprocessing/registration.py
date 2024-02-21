import os
import pickle
import argparse
from pathlib import Path
import deformetrica as dfca


def start_hippo_registration(data_path: Path, template_file: Path, output_path: Path):

    # instantiate a Deformetrica object
    deformetrica = dfca.Deformetrica(output_dir=str(output_path), verbosity="INFO")

    dataset_filenames = []
    subject_ids = []
    for sub_file in data_path.iterdir():
        assert sub_file.suffix == ".vtk", "File type must be .vtk"
        dataset_filenames.append([{"hippo": str(sub_file)}])
        subject_ids.append(sub_file.stem)

    dataset_specifications = {
        "dataset_filenames": dataset_filenames,
        "subject_ids": subject_ids,
    }

    template_specifications = {
        "hippo": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": "keops",
            "kernel_width": 0.03,
            "noise_std": 0.01,
            "filename": str(template_file),
            "attachment_type": "varifold",
            # "downsampling_factor": 1,
            # "dimension": "auto"
        }
    }
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": "auto",
        "max_iterations": 100,
        "initial_step_size": 0.5,
        "convergence_tolerance": 0.000001,
        "save_every_n_iters": 20,
        # "freeze_template": True,
        # "freeze_control_points": False
    }

    # perform a deterministic atlas estimation
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options={"deformation_kernel_type": "keops", "deformation_kernel_width": 0.05,
                       "number-of-timepoints": 25},
    )

    with open(output_path / 'dfca_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def start_box_registration(data_path: Path, template_file: Path, output_path: Path):

    # instantiate a Deformetrica object
    deformetrica = dfca.Deformetrica(output_dir=str(output_path), verbosity="INFO")

    dataset_filenames = []
    subject_ids = []
    for sub_file in data_path.iterdir():
        assert sub_file.suffix == ".vtk", "File type must be .vtk"
        dataset_filenames.append([{"box": str(sub_file)}])
        subject_ids.append(sub_file.stem)

    dataset_specifications = {
        "dataset_filenames": dataset_filenames,
        "subject_ids": subject_ids,
    }

    template_specifications = {
        "box": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": "keops",
            "kernel_width": 0.03,
            "noise_std": 0.01,
            "filename": str(template_file),
            "attachment_type": "varifold",
            # "downsampling_factor": 1,
            # "dimension": "auto"
        }
    }
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": "auto",
        "max_iterations": 100,
        "initial_step_size": 0.5,
        "convergence_tolerance": 0.000001,
        "save_every_n_iters": 20,
        # "freeze_template": True,
        # "freeze_control_points": False
    }

    # perform a deterministic atlas estimation
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options={"deformation_kernel_type": "keops", "deformation_kernel_width": 0.05,
                       "number-of-timepoints": 25},
    )

    with open(output_path / 'dfca_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def start_torus_registration(data_path: Path, template_file: Path, output_path: Path):

    # instantiate a Deformetrica object
    deformetrica = dfca.Deformetrica(output_dir=str(output_path), verbosity="INFO")

    dataset_filenames = []
    subject_ids = []
    for sub_file in data_path.iterdir():
        assert sub_file.suffix == ".vtk", "File type must be .vtk"
        dataset_filenames.append([{"torus": str(sub_file)}])
        subject_ids.append(sub_file.stem)

    dataset_specifications = {
        "dataset_filenames": dataset_filenames,
        "subject_ids": subject_ids,
    }

    template_specifications = {
        "torus": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": "keops",
            "kernel_width": 0.03,
            "noise_std": 0.01,
            "filename": str(template_file),
            "attachment_type": "varifold",
            # "downsampling_factor": 1,
            # "dimension": "auto"
        }
    }
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": "auto",
        "max_iterations": 100,
        "initial_step_size": 0.5,
        "convergence_tolerance": 0.000001,
        "save_every_n_iters": 20,
        # "freeze_template": True,
        # "freeze_control_points": False
    }

    # perform a deterministic atlas estimation
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options={"deformation_kernel_type": "keops", "deformation_kernel_width": 0.05,
                       "number-of-timepoints": 25},
    )

    with open(output_path / 'dfca_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Registration Pipeline')
    parser.add_argument('--use_case', type=str, default='hippocampus', choices=['hippocampus', 'box', 'torus'])
    parser.add_argument('--data_dir', type=str, default='hippo_data/')
    parser.add_argument('--template_file', type=str, default='hippo_template/ab300_283_standard.vtk')
    parser.add_argument('--output_dir', type=str, default='hippo_output/')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.use_case == 'hippocampus':
        start_hippo_registration(Path(args.data_dir),
                                 Path(args.template_file),
                                 Path(args.output_dir))

    elif args.use_case == 'box':
        start_box_registration(Path(args.data_dir),
                               Path(args.template_file),
                               Path(args.output_dir))

    elif args.use_case == 'torus':
        start_torus_registration(Path(args.data_dir),
                                 Path(args.template_file),
                                 Path(args.output_dir))
