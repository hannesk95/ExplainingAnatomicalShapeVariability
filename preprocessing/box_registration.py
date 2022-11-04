import deformetrica as dfca
from pathlib import Path
#from save_as_vtk_mesh import save_to_vtk
import pickle


def main(data_path: Path, template_file: Path, output_path: Path):

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
            #"downsampling_factor": 1,
            #"dimension": "auto"
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
        #"freeze_template": True,
        #"freeze_control_points": False
    }

    # perform a deterministic atlas estimation
    #model = deformetrica.estimate_registration(
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options={"deformation_kernel_type": "keops", "deformation_kernel_width": 0.05, "number-of-timepoints": 25},
    )

    with open(output_path / 'dfca_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main(
        Path("/home/jkiechle/scratch/box_data"),
        Path("/home/jkiechle/scratch/box_template/box.vtk"),
        Path("/home/jkiechle/scratch/box_output"),
    )
