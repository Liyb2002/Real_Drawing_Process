import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph

import Preprocessing.proc_CAD.generate_program
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper

import whole_process_helper.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper

import particle
# import whole_process_evaluate

from torch.utils.data import DataLoader
from tqdm import tqdm

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random
import copy
import re

# --------------------- Dataset --------------------- #
dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/small', return_data_path=True)


# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'program_output_dataset')



# --------------------- Compute_start_idx --------------------- #
def compute_start_idx():
    data_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    pattern = re.compile(r'.*_(\d+)$')
    
    largest_number = 0
    
    # List all directories and retrieve the number at the end of the format
    for d in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, d)):
            match = pattern.match(d)
            if match:
                number = int(match.group(1))  # Extract the number
                largest_number = max(largest_number, number)  # Keep track of the largest number

    
    return max(largest_number, 0)


# --------------------- Main Code --------------------- #
data_produced = compute_start_idx()
data_limit = 1
if os.path.exists(os.path.join(output_dir, f'data_{data_produced}')):
    shutil.rmtree(os.path.join(output_dir, f'data_{data_produced}'))
os.makedirs(os.path.join(output_dir, f'data_{data_produced}'), exist_ok=True)

print("data_produced", data_produced)

for data in tqdm(dataset, desc="Generating CAD Programs"):


    data_idx, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, data_path = data

    if data_produced >= data_limit:
        break
    
    
    print("working on:", data_path)

    cur_output_dir = os.path.join(output_dir, f'data_{data_produced}')
    if os.path.exists(cur_output_dir):
        shutil.rmtree(cur_output_dir)
    os.makedirs(cur_output_dir, exist_ok=True)
    

    base_particle = particle.Particle(data_produced, stroke_node_features, data_idx)
    base_particle.init_stroke_info(stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained)
    particle_list = []
    for particle_id in range (10):
        new_particle = copy.deepcopy(base_particle)
        new_particle.set_particle_id(particle_id, cur_output_dir)
        particle_list.append(new_particle)


    finished_particles = []
    while len(particle_list) > 0:
        # particle.next step 
        for cur_particle in particle_list:
            cur_particle.generate_next_step()

        # resample particles
        particle_list, finished_particles= whole_process_helper.helper.resample_particles(particle_list, finished_particles)

    # whole_process_helper.helper.find_top_different_particles(finished_particles, cur_output_dir)

    data_produced += 1
