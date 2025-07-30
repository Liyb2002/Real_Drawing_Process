import os
import json
import torch
from torch.utils.data import Dataset
import shutil
import re
import numpy as np
import pickle

import Preprocessing.cad2sketch_stroke_features


import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

import Encoders.helper
from tqdm import tqdm
from pathlib import Path
from deepdiff import DeepDiff

class cad2sketch_dataset_loader(Dataset):
    def __init__(self):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """

        self.data_path = os.path.join(os.getcwd(), 'dataset', 'small')

        self.load_dataset()


    def load_dataset(self):
        """
        Loads the dataset by iterating over all subfolders and storing their paths.
        """
        if not os.path.exists(self.data_path):
            print(f"Dataset path '{self.data_path}' not found.")
            return

        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        folders_sorted = sorted(folders, key=lambda x: int(os.path.basename(x)))

        if not folders:
            print("No folders found in the dataset directory.")
            return

        total_folders = 0
        target_index = 0
        for folder in folders_sorted:
            folder_index = int(os.path.basename(folder))
            if folder_index < target_index:
                continue
            
            total_folders += 1
            success_process = self.process_subfolder(os.path.join(self.data_path, folder))



    # IDEA:
    # We are in /selected_dataset/1600
    # Henro's code will create a /canvas folder that put all the .step files and the rotation matrix
    # process_subfolder() will give a /shape_info folder that stores all the .pkl files
    def process_subfolder(self, subfolder_path):
        """
        Processes an individual subfolder by reading JSON files and extracting relevant data.
        """

        data_idx = Path(subfolder_path).name
        print("data_idx", data_idx)

        # List all directories inside subfolder_path
        inner_dirs = [d for d in os.listdir(subfolder_path) 
                    if os.path.isdir(os.path.join(subfolder_path, d))]

        if len(inner_dirs) != 1:
            raise ValueError(f"Expected exactly one subfolder inside {subfolder_path}, found {len(inner_dirs)}")

        only_folder = os.path.join(subfolder_path, inner_dirs[0])
        final_edges_file_path = os.path.join(only_folder, 'final_edges.json')

        final_edges_data = self.read_json(final_edges_file_path)


        all_lines = Preprocessing.cad2sketch_stroke_features.extract_all_lines(final_edges_data)
        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines(all_lines)


        # -------------------------------------------------------------------------------- #


        stroke_node_features, is_feature_line_matrix= Preprocessing.cad2sketch_stroke_features.build_final_edges_json(all_lines)
        # Preprocessing.cad2sketch_stroke_features.vis_stroke_node_features(stroke_node_features)
        stroke_node_features = Preprocessing.cad2sketch_stroke_features.remove_duplicate (stroke_node_features)
        stroke_node_features, added_feature_lines= Preprocessing.cad2sketch_stroke_features.split_and_merge_stroke_cloud(stroke_node_features, is_feature_line_matrix)
       
        # print("len added_feature_lines", len(added_feature_lines))
        # Preprocessing.cad2sketch_stroke_features.vis_stroke_node_features_and_highlights(stroke_node_features, added_feature_lines)
        # Preprocessing.cad2sketch_stroke_features.vis_strokes_one_by_one(all_lines, stroke_node_features)
        # Preprocessing.cad2sketch_stroke_features.vis_circle_strokes(all_lines, stroke_node_features)


        # -------------------------------------------------------------------------------- #

        # Loops info
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.reorder_loops(stroke_cloud_loops)
        stroke_cloud_loops = [list(loop) for loop in stroke_cloud_loops]
        Preprocessing.cad2sketch_stroke_features.vis_feature_lines_loop_all(all_lines, stroke_node_features, stroke_cloud_loops)


        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features, loop_neighboring_all)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)


        # Stroke info
        connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
        strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)

        # Preprocessing.cad2sketch_stroke_features.vis_feature_lines_by_index_list(all_lines, stroke_node_features, [])

        for idxxx in [1]:
            connected_indices = np.where(strokes_perpendicular[idxxx] == 1)[0]
            Preprocessing.cad2sketch_stroke_features.vis_feature_lines_by_index_list(all_lines, stroke_node_features, [])
            Preprocessing.cad2sketch_stroke_features.vis_feature_lines_by_index_list(all_lines, stroke_node_features, connected_indices)

        output_file_path = os.path.join(subfolder_path, f'shape_info.pkl')
        with open(output_file_path, 'wb') as f:
            pickle.dump({
                'stroke_cloud_loops': stroke_cloud_loops, 

                'stroke_node_features': stroke_node_features,
                'strokes_perpendicular': strokes_perpendicular,

                'loop_neighboring_vertical': loop_neighboring_vertical,
                'loop_neighboring_horizontal': loop_neighboring_horizontal,
                'loop_neighboring_contained': loop_neighboring_contained,

            }, f)
            

  
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
