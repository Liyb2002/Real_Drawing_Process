import os
import json
import torch
from torch.utils.data import Dataset
import shutil
import re
import numpy as np
import pickle

import Preprocessing.cad2sketch_stroke_features
import Preprocessing.proc_CAD.perturbation_helper

from tqdm import tqdm
from pathlib import Path

class perturbation_dataset_loader(Dataset):
    def __init__(self, folder):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """

        self.data_path = os.path.join(os.getcwd(), 'dataset', folder)

        self.load_dataset()


    def load_dataset(self):
        final_edges_file_path = os.path.join(self.data_path, 'final_edges.json')

        final_edges_data = self.read_json(final_edges_file_path)
        all_lines = Preprocessing.cad2sketch_stroke_features.extract_all_lines(final_edges_data)


        stroke_node_features, _= Preprocessing.cad2sketch_stroke_features.build_final_edges_json(all_lines)
        # all_lines, stroke_node_features = Preprocessing.proc_CAD.perturbation_helper.remove_contained_lines(all_lines, stroke_node_features)
        # all_lines, stroke_node_features = Preprocessing.proc_CAD.perturbation_helper.duplicate_lines(all_lines, stroke_node_features)

        all_lines = Preprocessing.proc_CAD.perturbation_helper.compute_opacity(all_lines)

        perturbed_all_lines = Preprocessing.proc_CAD.perturbation_helper.do_perturb(all_lines, stroke_node_features)
        
        Preprocessing.cad2sketch_stroke_features.vis_feature_lines(perturbed_all_lines)

        perturbed_output_path = os.path.join(self.data_path, 'perturbed_all_lines.json')

        # Save to JSON file
        with open(perturbed_output_path, 'w') as f:
            json.dump(perturbed_all_lines, f, indent=4)


        return True




    def __getitem__(self, index):
        """
        Loads and processes the next subfolder when requested.
        If a subfolder has missing files, find the next available subfolder.
        Returns a tuple (intersection_matrix, all_edges_matrix, final_edges_matrix).
        """
        pass

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.subfolder_paths)

    def read_json(self, file_path):
        """
        Reads a JSON file and returns its contents.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None



# target = 'cad2sketch_annotated'
# d_generator = Preprocessing.perturb_stroke_cloud.perturbation_dataset_loader(target)
