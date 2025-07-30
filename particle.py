import Preprocessing.dataloader
import Preprocessing.generate_dataset_baseline
import Preprocessing.gnn_graph

import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.brep_read
import Preprocessing.proc_CAD.helper
import Preprocessing.cad2sketch_stroke_features

import whole_process_helper.helper

import Models.loop_embeddings

import Encoders.gnn.gnn
import Encoders.gnn_stroke.gnn
import Encoders.helper

import fidelity_score

from Preprocessing.config import device

from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import json
from pathlib import Path
import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import numpy as np
import random
import torch.nn.functional as F  

class Particle():
    def __init__(self, data_produced, stroke_node_features, data_idx):
        
        stroke_node_features = np.round(stroke_node_features, 4)
        self.stroke_node_features = stroke_node_features
        
        self.data_idx = data_idx
        self.gt_program = ['sketch', 'extrude', 'sketch', 'extrude', 'sketch', 'extrude', 'terminate']
        self.gt_sketch = [0,9,13]

        self.data_produced = data_produced

        self.brep_edges = torch.zeros(0)
        self.brep_loops = []
        self.cur__brep_class = Preprocessing.proc_CAD.generate_program.Brep()


        self.current_op = 1
        self.past_programs = [9]



        # Iteration infos
        self.selected_loop_indices = []

        # Particle State
        self.valid_particle = True
        self.success_terminate = False
        self.score = 1
        self.used_strokes = -1
        self.used_circle_strokes = -1

        self.true_value = 0.1


        # Feature_strokes
        self.predicted_feature_strokes = None


    def init_stroke_info(self, stroke_cloud_loops, strokes_perpendicular, loop_neighboring_vertical, loop_neighboring_horizontal, loop_neighboring_contained): 
        self.stroke_cloud_loops = stroke_cloud_loops
        
        self.strokes_perpendicular = strokes_perpendicular
        self.loop_neighboring_vertical = loop_neighboring_vertical
        self.loop_neighboring_horizontal = loop_neighboring_horizontal
        self.loop_neighboring_contained = loop_neighboring_contained


    def set_particle_id(self, particle_id, cur_output_dir_outerFolder):
        self.cur_output_dir = os.path.join(cur_output_dir_outerFolder, f'particle_{particle_id}')
        os.makedirs(self.cur_output_dir, exist_ok=True)
        
        self.particle_id = particle_id
        self.file_path = os.path.join(self.cur_output_dir, 'Program.json')


    def deepcopy_particle(self, new_id):

        new_particle = copy.copy(self)
        
        # manual copy, because we have tensors
        new_particle.brep_edges = self.brep_edges.copy()
        new_particle.brep_loops = self.brep_loops[:]
        new_particle.cur__brep_class = copy.deepcopy(self.cur__brep_class)
        new_particle.stroke_cloud_loops = copy.deepcopy(self.stroke_cloud_loops)
        new_particle.strokes_perpendicular = copy.deepcopy(self.strokes_perpendicular)
        new_particle.loop_neighboring_vertical = copy.deepcopy(self.loop_neighboring_vertical)
        new_particle.loop_neighboring_horizontal = copy.deepcopy(self.loop_neighboring_horizontal)
        new_particle.loop_neighboring_contained = copy.deepcopy(self.loop_neighboring_contained)
        new_particle.current_op = self.current_op
        new_particle.past_programs = self.past_programs[:]
        new_particle.selected_loop_indices = self.selected_loop_indices[:]
        new_particle.valid_particle = self.valid_particle
        new_particle.success_terminate = self.success_terminate
        new_particle.score = self.score
        new_particle.predicted_feature_strokes = (self.predicted_feature_strokes.clone() if self.predicted_feature_strokes is not None else None)
        new_particle.gt_program = self.gt_program
        new_particle.used_strokes = self.used_strokes


        cur_output_dir_outerFolder = os.path.dirname(self.cur_output_dir)
        new_folder_path = os.path.join(cur_output_dir_outerFolder, f'particle_{new_id}')
        shutil.copytree(self.cur_output_dir, new_folder_path)

        new_particle.particle_id = new_id
        new_particle.cur_output_dir = new_folder_path
        new_particle.file_path = os.path.join(new_folder_path, 'Program.json')
        new_particle.valid_particle = True

        return new_particle


    def set_gt_program(self, program):
        self.gt_program = program


    def program_terminated(self, gnn_graph):
        
        if (len(self.gt_program) == len(self.past_programs)):
            stroke_features_file = os.path.join(self.cur_output_dir, 'stroke_cloud_features.json')
            stroke_features_list = self.stroke_node_features.tolist()

            with open(stroke_features_file, 'w') as json_file:
                for stroke in stroke_features_list:
                    json.dump(stroke, json_file)
                    json_file.write("\n")

        return len(self.gt_program) == len(self.past_programs)
        


    def particle_score(self):
        return self.score
    

    def is_valid_particle(self):
        return self.valid_particle


    def mark_off_new_strokes(self, stroke_to_brep):

        new_strokes_mark_off = np.sum(stroke_to_brep == 1)

        if len(self.past_programs) == 1:
            self.used_strokes = new_strokes_mark_off
            return True
        
        if self.used_strokes < new_strokes_mark_off:
            self.used_strokes = new_strokes_mark_off
            return True
        
        return False

        

    def generate_next_step(self):
        
        try:
            stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(self.stroke_node_features, self.brep_edges)
            stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(self.stroke_node_features, self.brep_edges)
            stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)
            stroke_to_loop = Preprocessing.cad2sketch_stroke_features.from_stroke_to_edge(stroke_to_edge, self.stroke_cloud_loops)

            # print("used strokes", np.where(stroke_to_edge_lines[:, 0] == 1)[0])
            # 2) Build graph
            gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
                self.stroke_cloud_loops, 
                self.stroke_node_features, 
                self.strokes_perpendicular, 
                self.loop_neighboring_vertical, 
                self.loop_neighboring_horizontal, 
                self.loop_neighboring_contained,
                stroke_to_loop,
                stroke_to_edge
            )


            # Encoders.helper.vis_all_loops(gnn_graph['stroke'].x.cpu().numpy(), self.data_idx, self.stroke_cloud_loops)
            
            new_mark_off = self.mark_off_new_strokes(stroke_to_edge)

            # if len(self.past_programs) == 5:
            #     Encoders.helper.vis_brep(self.brep_edges)
            #     used_indices = np.where(stroke_to_edge > 0.5)[0].tolist()
            #     Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), used_indices, self.data_idx)

            if not new_mark_off:
                self.correct_termination_check()
                return



            # compute particle score
            self.fidelity_score = do_fidelity_score_prediction(gnn_graph)
            # print("predicted fidelity_score", self.fidelity_score)


            if self.current_op == 1:
                print("Build sketch")

                num_existing_sketches = self.past_programs.count(1)
                next_sketch_idx = self.gt_sketch[num_existing_sketches]

                self.sketch_selection_mask, self.sketch_points, normal, selected_loop_idx, prob = do_sketch(gnn_graph, self.data_idx, next_sketch_idx)
                self.selected_loop_indices.append(selected_loop_idx)
                self.score = self.score * prob

                tmpt_brep_class = Preprocessing.proc_CAD.generate_program.Brep()
                if self.sketch_points.shape[0] == 1:
                    # do circle sketch
                    self.cur__brep_class.regular_sketch_circle(self.sketch_points[0, 3:6].tolist(), self.sketch_points[0, 7].item(), self.sketch_points[0, :3].tolist())
                    tmpt_brep_class.regular_sketch_circle(self.sketch_points[0, 3:6].tolist(), self.sketch_points[0, 7].item(), self.sketch_points[0, :3].tolist())
                else: 
                    self.cur__brep_class._sketch_op(self.sketch_points, normal, self.sketch_points)
                    tmpt_brep_class._sketch_op(self.sketch_points, normal, self.sketch_points)

                tmpt_brep_class.write_to_json(self.cur_output_dir, True)
                tmpt_file_path = os.path.join(self.cur_output_dir, 'tempt_Program.json')
                tmpt_parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(tmpt_file_path, self.cur_output_dir)
                tmpt_parsed_program_class.read_json_file(len(self.past_programs))


            # Build Extrude
            if self.current_op == 2:
                print("Build extrude")

                self.extrude_amount, self.extrude_direction, prob = do_extrude(gnn_graph, self.sketch_selection_mask, self.sketch_points, self.brep_edges, self.data_idx)
                
                prob = 1.0
                self.cur__brep_class.extrude_op(self.extrude_amount.item(), self.extrude_direction.detach().cpu().numpy())
                self.score = self.score * prob


            # Build fillet
            if self.current_op == 3:
                print("Build Fillet")
                output_fillet_edge, selected_prob = do_fillet(gnn_graph, self.brep_edges, self.data_idx)
                self.cur__brep_class.random_fillet(output_fillet_edge)
                self.score = self.score * selected_prob


            if self.current_op ==4:
                print("Build Chamfer")
                chamfer_edge, chamfer_amount, prob= do_chamfer(gnn_graph, self.brep_edges)
                self.cur__brep_class.random_chamfer(chamfer_edge, chamfer_amount)
                self.score = self.score * prob


            # 5.3) Write to brep
            self.cur__brep_class.write_to_json(self.cur_output_dir)


            # 5.4) Read the program and produce the brep file
            parsed_program_class = Preprocessing.proc_CAD.Program_to_STL.parsed_program(self.file_path, self.cur_output_dir)
            parsed_program_class.read_json_file()


            # 5.5) Read brep file
            cur_relative_output_dir = os.path.join(output_dir_name, f'data_{self.data_produced}', f'particle_{self.particle_id}')

            brep_files = [file_name for file_name in os.listdir(os.path.join(cur_relative_output_dir, 'canvas'))
                    if file_name.endswith('.step')]
            brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


            # 5.6) Update brep data
            prev_brep_edges = self.brep_edges
            brep_path = os.path.join(output_dir_name, f'data_{self.data_produced}', f'particle_{self.particle_id}', 'canvas')
            self.brep_edges, self.brep_loops = cascade_brep(brep_files, self.data_produced, brep_path)
            # self.brep_loops = Preprocessing.proc_CAD.helper.remove_duplicate_circle_breps(self.brep_loops, self.brep_edges)

            # Compute Chamfer Distance
            # cur_fidelity_score = fidelity_score.compute_fidelity_score(self.gt_brep_file_path, os.path.join(brep_path, brep_files[-1]))
            cur_fidelity_score = -1

            self.past_programs.append(self.current_op)
            # self.current_op, op_prob = program_prediction(gnn_graph, self.past_programs)
            # self.score = self.score * op_prob

            # Start hack -------------------------------- #
            next_op = self.gt_program[len(self.past_programs) - 1]
            if next_op == 'sketch':
                self.current_op = 1
            elif next_op == 'extrude':
                self.current_op = 2
            elif next_op == 'fillet':
                self.current_op = 3
            elif next_op == 'chamfer':
                self.current_op = 4
            elif next_op == 'terminate':
                self.current_op = 0
            # End hack -------------------------------- #

            print("----------------")
            print("self.past_programs", self.past_programs)
            print("self.gt_program", self.gt_program)
            print("self.current_op", self.current_op)

            # 6) Write the stroke_cloud data to pkl file
            output_file_path = os.path.join(self.cur_output_dir, 'canvas', f'{len(brep_files)-1}_eval_info.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump({
                    'stroke_node_features': self.stroke_node_features,
                    'cur_fidelity_score' : cur_fidelity_score, 

                    'stroke_cloud_loops': self.stroke_cloud_loops, 

                    'stroke_node_features': self.stroke_node_features,
                    'strokes_perpendicular': self.strokes_perpendicular,

                    'loop_neighboring_vertical': self.loop_neighboring_vertical,
                    'loop_neighboring_horizontal': self.loop_neighboring_horizontal,
                    'loop_neighboring_contained': self.loop_neighboring_contained,

                    'stroke_to_loop': stroke_to_loop,
                    'stroke_to_edge': stroke_to_edge

                }, f)
            

                
        except Exception as e:
            print(f"An error occurred: {e}")
            self.valid_particle = False

            self.correct_termination_check()
        
            
            if len(self.past_programs) > 9:
                self.success_terminate = True
            else:
                self.remove_particle()


        if self.current_op == 0:
            # do a final check
            self.success_terminate = True
            self.correct_termination_check()
            return 



    def remove_particle(self):
        if os.path.exists(self.cur_output_dir):
            shutil.rmtree(self.cur_output_dir)


    def correct_termination_check(self):
        if len(self.past_programs) > 5:
            self.success_terminate = True
            self.valid_particle = False 
            return 


        self.remove_particle()
        self.valid_particle = False

        return 




# ---------------------------------------------------------------------------------------------------------------------------------- #



# --------------------- Directory --------------------- #
current_dir = os.getcwd()
output_dir_name = 'program_output_dataset'
output_dir = os.path.join(current_dir, output_dir_name)


# --------------------- Skecth Network --------------------- #
sketch_graph_encoder = Encoders.gnn.gnn.SemanticModule()
sketch_graph_decoder = Encoders.gnn.gnn.Sketch_Decoder()
sketch_graph_encoder.eval()
sketch_graph_decoder.eval()
sketch_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction')
# sketch_dir = os.path.join(current_dir, 'checkpoints', 'sketch_prediction_synthetic')
sketch_graph_encoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_encoder.pth'), weights_only=True))
sketch_graph_decoder.load_state_dict(torch.load(os.path.join(sketch_dir, 'graph_decoder.pth'), weights_only=True))

def predict_sketch(gnn_graph, data_idx, next_sketch_idx):
        
    x_dict = sketch_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    sketch_selection_mask = sketch_graph_decoder(x_dict)

    next_mask = torch.zeros_like(sketch_selection_mask)
    next_mask[next_sketch_idx] = 1
    sketch_selection_mask= next_mask



    selected_loop_idx, idx_prob = whole_process_helper.helper.find_valid_sketch(gnn_graph, sketch_selection_mask)
    sketch_stroke_idx = Encoders.helper.find_selected_strokes_from_loops(gnn_graph['stroke', 'represents', 'loop'].edge_index, selected_loop_idx)
    # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), sketch_stroke_idx, data_idx)

    return selected_loop_idx, sketch_selection_mask, idx_prob

def do_sketch(gnn_graph, data_idx, next_sketch_idx):
    selected_loop_idx, sketch_selection_mask, idx_prob= predict_sketch(gnn_graph, data_idx, next_sketch_idx)
    sketch_points = whole_process_helper.helper.extract_unique_points(selected_loop_idx[0], gnn_graph)
    normal = [1, 0, 0]
    sketch_selection_mask = whole_process_helper.helper.clean_mask(sketch_selection_mask, selected_loop_idx)
    return sketch_selection_mask, sketch_points, normal, selected_loop_idx, idx_prob


# --------------------- Extrude Network --------------------- #
extrude_graph_encoder = Encoders.gnn.gnn.SemanticModule()
extrude_graph_decoder = Encoders.gnn.gnn.Extrude_Decoder()
extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
# extrude_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction_synthetic')
extrude_graph_encoder.eval()
extrude_graph_decoder.eval()
extrude_graph_encoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_encoder.pth'), weights_only=True))
extrude_graph_decoder.load_state_dict(torch.load(os.path.join(extrude_dir, 'graph_decoder.pth'), weights_only=True))


extrude_face_graph_encoder = Encoders.gnn.gnn.SemanticModule()
extrude_face_graph_decoder = Encoders.gnn.gnn.Extruded_Face_Decoder()
extrude_face_dir = os.path.join(current_dir, 'checkpoints', 'extruded_face_prediction')
# extrude_face_dir = os.path.join(current_dir, 'checkpoints', 'extruded_face_prediction_synthetic')
extrude_face_graph_encoder.eval()
extrude_face_graph_decoder.eval()
extrude_face_graph_encoder.load_state_dict(torch.load(os.path.join(extrude_face_dir, 'graph_encoder.pth'), weights_only=True))
extrude_face_graph_decoder.load_state_dict(torch.load(os.path.join(extrude_face_dir, 'graph_decoder.pth'), weights_only=True))



# def predict_extrude(gnn_graph, sketch_selection_mask, data_idx):
#     gnn_graph.set_select_sketch(sketch_selection_mask)

#     x_dict = extrude_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
#     extrude_selection_mask = extrude_graph_decoder(x_dict)
    
#     extrude_stroke_idx =  (extrude_selection_mask >= 0.5).nonzero(as_tuple=True)[0]
#     _, extrude_stroke_idx = torch.max(extrude_selection_mask, dim=0)
#     # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_stroke_idx, data_idx)
#     return extrude_selection_mask

# def do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges, data_idx):
#     extrude_selection_mask = predict_extrude(gnn_graph, sketch_selection_mask, data_idx)
#     extrude_amount, extrude_direction, selected_prob= whole_process_helper.helper.get_extrude_amount(gnn_graph, extrude_selection_mask, sketch_points, brep_edges)
#     return extrude_amount, extrude_direction, selected_prob



def do_extrude(gnn_graph, sketch_selection_mask, sketch_points, brep_edges, data_idx):

    extrude_amount, extrude_direction, prob = whole_process_helper.helper.get_extrude_amount_fallback(gnn_graph, sketch_points)
    return extrude_amount, extrude_direction, prob





# --------------------- Fillet Network --------------------- #
fillet_graph_encoder = Encoders.gnn.gnn.SemanticModule()
fillet_graph_decoder = Encoders.gnn.gnn.Fillet_Decoder()
fillet_dir = os.path.join(current_dir, 'checkpoints', 'fillet_prediction')
# fillet_dir = os.path.join(current_dir, 'checkpoints', 'fillet_prediction_synthetic')
fillet_graph_encoder.eval()
fillet_graph_decoder.eval()
fillet_graph_encoder.load_state_dict(torch.load(os.path.join(fillet_dir, 'graph_encoder.pth'), weights_only=True))
fillet_graph_decoder.load_state_dict(torch.load(os.path.join(fillet_dir, 'graph_decoder.pth'), weights_only=True))


def do_fillet(gnn_graph, brep_edges, data_idx):
    # output_fillet_edge, selected_prob = whole_process_helper.helper.get_output_fillet_edge(gnn_graph, fillet_selection_mask)

    output_fillet_edge, selected_prob = whole_process_helper.helper.get_output_fillet_edge_fallback(gnn_graph)

    return output_fillet_edge, selected_prob





# --------------------- Chamfer Network --------------------- #
chamfer_graph_encoder = Encoders.gnn.gnn.SemanticModule()
chamfer_graph_decoder = Encoders.gnn.gnn.Chamfer_Decoder()
# chanfer_dir = os.path.join(current_dir, 'checkpoints', 'chamfer_prediction')
chanfer_dir = os.path.join(current_dir, 'checkpoints', 'chamfer_prediction_synthetic')
chamfer_graph_encoder.eval()
chamfer_graph_decoder.eval()
chamfer_graph_encoder.load_state_dict(torch.load(os.path.join(chanfer_dir, 'graph_encoder.pth'), weights_only=True))
chamfer_graph_decoder.load_state_dict(torch.load(os.path.join(chanfer_dir, 'graph_decoder.pth'), weights_only=True))


def predict_chamfer(gnn_graph):
    # gnn_graph.padding()
    x_dict = chamfer_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    chamfer_selection_mask = chamfer_graph_decoder(x_dict)

    # print("gnn_graph['stroke'].x", gnn_graph['stroke'].x.shape)

    # chamfer_stroke_idx =  (chamfer_selection_mask >= 0.3).nonzero(as_tuple=True)[0]
    # _, chamfer_stroke_idx = torch.topk(chamfer_selection_mask.flatten(), k=2)
    _, chamfer_stroke_idx = torch.max(chamfer_selection_mask, dim=0)
    # Encoders.helper.vis_selected_strokes_synthetic(gnn_graph['stroke'].x.cpu().numpy(), chamfer_stroke_idx)
    
    return chamfer_selection_mask


def do_chamfer(gnn_graph, brep_edges):
    chamfer_selection_mask = predict_chamfer(gnn_graph)
    chamfer_edge, chamfer_amount, selected_prob= whole_process_helper.helper.get_chamfer_amount(gnn_graph, chamfer_selection_mask, brep_edges)
    return chamfer_edge, chamfer_amount.item(), selected_prob




# --------------------- Operation Prediction Network --------------------- #
operation_graph_encoder = Encoders.gnn.gnn.SemanticModule()
operation_graph_decoder= Encoders.gnn.gnn.Program_Decoder()
program_dir = os.path.join(current_dir, 'checkpoints', 'operation_prediction')
operation_graph_encoder.eval()
operation_graph_decoder.eval()
operation_graph_encoder.load_state_dict(torch.load(os.path.join(program_dir, 'graph_encoder.pth'), weights_only=True))
operation_graph_decoder.load_state_dict(torch.load(os.path.join(program_dir, 'graph_decoder.pth'), weights_only=True))


def program_prediction(gnn_graph, past_programs):
    past_programs = whole_process_helper.helper.padd_program(past_programs)
    gnn_graph.padding()
    x_dict = operation_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output = operation_graph_decoder(x_dict, past_programs)

    predicted_class, class_prob = whole_process_helper.helper.sample_operation(output)
    return predicted_class, class_prob



# --------------------- Fidelity Score --------------------- #
fidelity_graph_encoder = Encoders.gnn.gnn.SemanticModule()
fidelity_graph_decoder= Encoders.gnn.gnn.Fidelity_Decoder()
fidelity_dir = os.path.join(current_dir, 'checkpoints', 'fidelity_prediction')
fidelity_graph_encoder.eval()
fidelity_graph_decoder.eval()
fidelity_graph_encoder.load_state_dict(torch.load(os.path.join(fidelity_dir, 'graph_encoder.pth'), weights_only=True))
fidelity_graph_decoder.load_state_dict(torch.load(os.path.join(fidelity_dir, 'graph_decoder.pth'), weights_only=True))


def do_fidelity_score_prediction(gnn_graph):
    x_dict = fidelity_graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
    output_logits = fidelity_graph_decoder(x_dict, True)
    predicted_bin = torch.argmax(output_logits, dim=1)
    return predicted_bin.item()


# --------------------- Cascade Brep Features --------------------- #

def cascade_brep(brep_files, data_produced, brep_path):

    final_brep_edges = []
    final_cylinder_features = []

    for file_name in brep_files:
        brep_file_path = os.path.join(brep_path, file_name)
        edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        
        if len(final_brep_edges) == 0:
            final_brep_edges = edge_features_list
            final_cylinder_features = cylinder_features
        else:
            # We already have brep
            new_features = Preprocessing.generate_dataset_baseline.find_new_features(final_brep_edges, edge_features_list) 
            final_brep_edges += new_features
            final_cylinder_features += cylinder_features

    output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
    brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(output_brep_edges) + Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)
    brep_loops = [list(loop) for loop in brep_loops]
    return output_brep_edges, brep_loops



def get_final_brep(brep_path, last_file):
    
    brep_file_path = os.path.join(brep_path, last_file)
    edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
    return edge_features_list