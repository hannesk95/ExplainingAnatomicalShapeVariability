import os
import glob
import math
import shutil
import argparse
import subprocess
import numpy as np
import shapeworks as sw

from tqdm import tqdm


def run_box_pipeline(args):

    mesh_files = sorted(glob.glob("box_raw_data/*.ply"))[::1]

    ###################################################################################################################

    print("Sort meshes into batches")
    # # Load meshes
    # meshes = []
    # for mesh_file in tqdm(mesh_files):
    #     meshes.append(sw.Mesh(mesh_file))
    # # Get distances
    # print("Sorting based on surface-to-surface distance...")
    # distances = np.zeros(len(meshes))
    # ref_index = sw.find_reference_mesh_index(meshes)
    # ref_mesh = meshes[ref_index]
    # for i in tqdm(range(len(meshes))):
    #     distances[i] = np.mean(meshes[i].distance(ref_mesh)[0])
    # # Sort
    # sorted_indices = np.argsort(distances)
    # sorted_mesh_files = np.array(mesh_files)[sorted_indices]
    # Make batches
    batch_size = math.ceil(len(mesh_files) / 10)
    batches = [mesh_files[i:i + batch_size] for i in range(0, len(mesh_files), batch_size)]
    print("Created " + str(len(batches)) + " batches of size " + str(len(batches[0])))

    ###################################################################################################################

    print("Optimize initial particles on first batch")
    # Create project spreadsheet
    project_location = "box_shape_model/"
    if not os.path.exists(project_location):
        os.makedirs(project_location)
    # Remove particle dir if it already exists
    shape_model_dir = project_location + 'box_particles/'
    if os.path.exists(shape_model_dir):
        shutil.rmtree(shape_model_dir)
    # Set subjects
    subjects = []
    number_domains = 1
    for i in range(len(batches[0])):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[0][i]], project_location)
        subject.set_original_filenames(rel_mesh_file)
        subject.set_groomed_filenames(rel_mesh_file)
        subjects.append(subject)
    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    # Create a dictionary for all the parameters required by optimization
    parameter_dictionary = {
        "number_of_particles": args.particles,
        "use_normals": 0,
        "normal_weight": 10.0,
        "checkpointing_interval": 300,
        "keep_checkpoints": 0,
        "iterations_per_split": 300,
        "optimization_iterations": 300,
        "starting_regularization": 10,
        "ending_regularization": 1,
        "recompute_regularization_interval": 1,
        "domains_per_shape": 1,
        "relative_weighting": 10,
        "initial_relative_weighting": 0.05,
        "procrustes_interval": 0,
        "procrustes_scaling": 0,
        "save_init_splits": 0,
        "verbosity": 0
    }

    # Run multiscale optimization unless single scale is specified
    # if not args.use_single_scale:
    #     parameter_dictionary["multiscale"] = 1
    #     parameter_dictionary["multiscale_particles"] = 32
    # If running a tiny test, reduce some parameters
    # if args.tiny_test:
    #     parameter_dictionary["number_of_particles"] = 8
    #     parameter_dictionary["optimization_iterations"] = 1
    #     parameter_dictionary["starting_regularization"] = 10000
    #     parameter_dictionary["ending_regularization"] = 1000

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key, sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant('mesh'))
    project.set_parameters("optimize", parameters)
    spreadsheet_file = "box_shape_model/box.xlsx"
    project.save(spreadsheet_file)

    # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    print(optimize_cmd)
    subprocess.check_call(optimize_cmd)

    ###################################################################################################################

    if args.full_data:
        print("Incremental optimization")

        # Update parameters for incremental optimization
        parameter_dictionary["use_landmarks"] = 1  # For particle initialization
        parameter_dictionary["iterations_per_split"] = 0  # No initialization iterations
        parameter_dictionary["optimization_iterations"] = 100  # Fewer optimization iterations
        parameter_dictionary["multiscale"] = 0  # Single scale

        # if args.tiny_test:
        #     parameter_dictionary["number_of_particles"] = 8
        #     parameter_dictionary["optimization_iterations"] = 1
        #     parameter_dictionary["starting_regularization"] = 10000
        #     parameter_dictionary["ending_regularization"] = 1000

        # Run optimization on each batch
        for batch_index in range(1, len(batches)):
            print("Running incremental optimization " + str(batch_index) + " out of " + str(len(batches) - 1))
            # Update meanshape
            sw.utils.findMeanShape(shape_model_dir)
            mean_shape_path = shape_model_dir + '/meanshape_local.particles'
            # Set subjects
            subjects = []
            # Add current shape model (e.g. all previous batches)
            for i in range(0, batch_index):
                for j in range(len(batches[i])):
                    subject = sw.Subject()
                    subject.set_number_of_domains(1)
                    rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[i][j]], project_location)
                    subject.set_original_filenames(rel_mesh_file)
                    subject.set_groomed_filenames(rel_mesh_file)
                    particle_file = shape_model_dir + os.path.basename(rel_mesh_file[0]).replace(".ply", "_local.particles")
                    rel_particle_file = sw.utils.get_relative_paths([os.getcwd() + "/" + particle_file], project_location)
                    subject.set_landmarks_filenames(rel_particle_file)
                    subjects.append(subject)
            # Add new shapes in current batch - intialize with meanshape
            for j in range(len(batches[batch_index])):
                subject = sw.Subject()
                subject.set_number_of_domains(1)
                rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[batch_index][j]], project_location)
                subject.set_original_filenames(rel_mesh_file)
                subject.set_groomed_filenames(rel_mesh_file)
                rel_particle_file = sw.utils.get_relative_paths([os.getcwd() + "/" + mean_shape_path], project_location)
                subject.set_landmarks_filenames(rel_particle_file)
                subjects.append(subject)
            # Set project
            project = sw.Project()
            project.set_subjects(subjects)
            parameters = sw.Parameters()

            # Add param dictionary to spreadsheet
            for key in parameter_dictionary:
                parameters.set(key, sw.Variant([parameter_dictionary[key]]))
            parameters.set("domain_type", sw.Variant('mesh'))
            project.set_parameters("optimize", parameters)
            spreadsheet_file = "box_shape_model/box.xlsx"
            project.save(spreadsheet_file)

            # Run optimization
            optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
            subprocess.check_call(optimize_cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ShapeWorks Box Pipeline')
    parser.add_argument('--particles', type=int, default=128)
    parser.add_argument('--full_data', type=bool, default=False)
    args = parser.parse_args()

    run_box_pipeline(args)
