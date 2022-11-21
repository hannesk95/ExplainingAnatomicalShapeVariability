import os
import argparse
import subprocess
import numpy as np
import shapeworks as sw

from glob import glob
from tqdm import tqdm


def preprocessing(args):
    """Preprocess data."""
    # data_dir = 'data/'
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)

    for segmentation in tqdm(glob(f"{args.data_dir}*.nii")):
        filename = segmentation.split("\\")[-1].split(".")[0].replace("_standard", "")

        # Read binary segmentation and split into left and right
        seg = sw.Image(segmentation)
        seg_left = seg.binarize(1.5, 2.5, 1)
        seg = sw.Image(segmentation)
        seg_right = seg.binarize(0.5, 1.5, 1)

        # Padding
        seg_left.pad(5, 0)
        seg_right.pad(5, 0)

        # Compute distance transform
        seg_left.computeDT(0)
        seg_right.computeDT(0)

        # Reduce high-frequency information
        seg_left.gaussianBlur(0.05)
        seg_right.gaussianBlur(0.05)

        # Convert into mesh
        mesh_left = seg_left.toMesh(1)
        mesh_right = seg_right.toMesh(1)

        # Smooth mesh surface
        mesh_left.smooth(3, 1)
        mesh_right.smooth(3, 1)

        mesh_left.write(f"{args.output_dir}{filename}_left.vtk")
        mesh_right.write(f"{args.output_dir}{filename}_right.vtk")


def grooming(args):
    """Groom data."""
    # groom_dir = 'groomed/'
    # if not os.path.exists(groom_dir):
    #     os.makedirs(groom_dir)

    groom_dir = args.output_dir

    mesh_files = glob(f"{args.output_dir}*.vtk")

    # list of shape segmentations
    mesh_list = []
    # list of shape names (shape files prefixes) to be used for saving outputs
    mesh_names = []
    domain_ids = []
    for mesh_file in mesh_files:
        print('Loading: ' + mesh_file)
        # get current shape name
        mesh_name = mesh_file.split('\\')[-1].replace('.vtk', '')
        mesh_names.append(mesh_name)
        # get domain identifiers
        domain_ids.append(mesh_name.split(".")[0].split("_")[-1])

        # load mesh
        mesh = sw.Mesh(mesh_file)
        # append to the mesh list
        mesh_list.append(mesh)

    # domain identifiers for all shapes
    domain_ids = np.array(domain_ids)
    # shape index for all shapes in domain 1
    domain1_indx = list(np.where(domain_ids == 'left')[0])
    # shape index for all shapes in domain 2
    domain2_indx = list(np.where(domain_ids == 'right')[0])

    # Select reference for rigid alignment
    print("[INFO] Looking for reference (medoid) shape ...")
    domains_per_shape = 2
    domain_1_meshes = []
    # get domain 1 shapes
    for i in range(int(len(mesh_list) / domains_per_shape)):
        domain_1_meshes.append(mesh_list[i * domains_per_shape])

    ref_index = sw.find_reference_mesh_index(domain_1_meshes)
    domain1_reference = mesh_list[ref_index * domains_per_shape].copy()
    domain2_reference = mesh_list[ref_index * domains_per_shape + 1].copy()
    domain1_ref_name = mesh_names[ref_index * domains_per_shape]
    domain2_ref_name = mesh_names[ref_index * domains_per_shape + 1]
    reference = [domain1_reference, domain2_reference]
    ref_name = [domain1_ref_name, domain2_ref_name]

    # Rigid alignment
    transforms = []
    for i in range(len(domain_1_meshes)):

        # calculate the transformation
        for d in range(domains_per_shape):
            # compute rigid transformation
            rigidTransform = mesh_list[i * domains_per_shape + d].createTransform(reference[d],
                                                                                  sw.Mesh.AlignmentType.Rigid, 100)
            name = mesh_names[i * domains_per_shape + d]
            print('Aligning ' + name + ' to ' + ref_name[d])
            transforms.append(rigidTransform)

    # Delete old groomed meshes
    delete_meshes = glob(f"{args.output_dir}*.vtk")
    for file in delete_meshes:
        os.remove(file)

    # Save groomed meshes
    groomed_mesh_files = sw.utils.save_meshes(groom_dir, mesh_list, mesh_names, extension='vtk')

    return domain_1_meshes, mesh_files, groomed_mesh_files, transforms


def particle_optimization(args, domain_1_meshes, mesh_files, groomed_mesh_files, transforms):
    # Create project spreadsheet
    if not os.path.exists(args.project_location):
        os.makedirs(args.project_location)

    # Set subjects
    subjects = []

    for i in range(len(domain_1_meshes)):
        subject = sw.Subject()
        subject.set_number_of_domains(args.domains_per_shape)
        rel_mesh_files = []
        rel_groom_files = []
        transform = []
        for d in range(args.domains_per_shape):
            rel_mesh_files += sw.utils.get_relative_paths([os.getcwd() + '/' + mesh_files[i * args.domains_per_shape + d]],
                                                          args.project_location)
            rel_groom_files += sw.utils.get_relative_paths(
                [os.getcwd() + '/' + groomed_mesh_files[i * args.domains_per_shape + d]], args.project_location)
            transform.append(transforms[i * args.domains_per_shape + d].flatten())
        subject.set_groomed_transforms(transform)
        subject.set_groomed_filenames(rel_groom_files)
        subject.set_original_filenames(rel_mesh_files)
        subjects.append(subject)

    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    parameter_dictionary = {
        "checkpointing_interval": 200,
        "keep_checkpoints": 0,
        "iterations_per_split": 200,
        "optimization_iterations": 200,
        "starting_regularization": 1000,
        "ending_regularization": 0.1,
        "recompute_regularization_interval": 1,
        "domains_per_shape": args.domains_per_shape,
        "relative_weighting": 10,
        "initial_relative_weighting": 0.1,
        "procrustes_interval": 0,
        "procrustes_scaling": 0,
        "save_init_splits": 0,
        "verbosity": 1

    }
    num_particles = [args.particles, args.particles]

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key, sw.Variant([parameter_dictionary[key]]))
    parameters.set("number_of_particles", sw.Variant(num_particles))
    project.set_parameters("optimize", parameters)

    spreadsheet_file = f"{args.project_location}/{args.spreadsheet_name}"
    project.save(spreadsheet_file)

    # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)


def run_studio():
    """Open ShapeWorks Studio."""
    spreadsheet_file = f"{args.project_location}/{args.spreadsheet_name}"
    analyze_cmd = ('ShapeWorksStudio ' + spreadsheet_file).split()
    subprocess.check_call(analyze_cmd)


def run_shapeworks_pipeline(args) -> None:

    preprocessing(args)
    domain_1_meshes, mesh_files, groomed_mesh_files, transforms = grooming(args)

    particles_list = [2048, 4096]
    for particles in particles_list:
        args.project_location = f"{args.project_location}_{str(particles)}"
        args.particles = particles
        particle_optimization(args, domain_1_meshes, mesh_files, groomed_mesh_files, transforms)

    if args.studio:
        run_studio()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ShapeWorks Pipeline')
    parser.add_argument('--data_dir', type=str, default='binary_segmentations/')
    parser.add_argument('--output_dir', type=str, default='groomed/')
    parser.add_argument('--project_location', type=str, default='shape_model')
    parser.add_argument('--spreadsheet_name', type=str, default='hippocampus.xlsx')
    parser.add_argument('--particles', type=int, default=128)
    parser.add_argument('--studio', type=bool, default=False)
    parser.add_argument('--studio_only', type=bool, default=False)
    parser.add_argument('--domains_per_shape', type=int, default=2)
    args = parser.parse_args()

    #args.project_location = f"{args.project_location}_{str(args.particles)}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.studio_only:
        run_studio()
    else:
        run_shapeworks_pipeline(args)
