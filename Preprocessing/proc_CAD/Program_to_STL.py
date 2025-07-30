import json
import os
from copy import deepcopy

import numpy as np
import Preprocessing.proc_CAD.build123.protocol
import Preprocessing.proc_CAD.helper

import random

class parsed_program():
    def __init__(self, file_path, data_directory = None, output = True):
        self.file_path = file_path
        self.data_directory = data_directory

        if not data_directory:
            self.data_directory = os.path.dirname(__file__)

        canvas_directory = os.path.join(self.data_directory, 'canvas')
        os.makedirs(canvas_directory, exist_ok=True)

        self.canvas = None
        self.prev_sketch = None
        self.Op_idx = 0
        self.output = output
        
    def read_json_file(self, tempt_idx = 0):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            self.len_program = len(data)
            for i in range(len(data)):
                Op = data[i]
                operation = Op['operation']
                
                if operation[0] == 'sketch':
                    self.parse_sketch(Op, tempt_idx)
                
                if operation[0] == 'extrude':
                    self.parse_extrude(Op, data[i-1])
                
                if operation[0] == 'fillet':
                    self.parse_fillet(Op)
                
                if operation[0] == 'chamfer':
                    self.parse_chamfer(Op)
 
                if operation[0] == 'terminate':
                    self.Op_idx += 1
                    break

        return
            

    def parse_sketch(self, Op, tempt_idx):
        if 'radius' in Op['faces'][0]:
            self.parse_circle(Op)
            return 

        point_list = [vert['coordinates'] for vert in Op['vertices']]
        
        new_point_list = [point_list[0]]  # Start with the first point
        for i in range(1, len(point_list)):
            # Append each subsequent point twice
            new_point_list.append(point_list[i])
            new_point_list.append(point_list[i])
        
        # Add the first point again at the end to close the loop
        new_point_list.append(point_list[0])
        if tempt_idx == 0:
            self.prev_sketch = Preprocessing.proc_CAD.build123.protocol.build_sketch(self.Op_idx, self.canvas, new_point_list, self.output, self.data_directory)
        else:
            Preprocessing.proc_CAD.build123.protocol.build_sketch(self.Op_idx, self.canvas, new_point_list, self.output, self.data_directory, tempt_idx)
        self.Op_idx += 1

        



    def parse_circle(self, Op):
        radius = Op['faces'][0]['radius']
        center = Op['faces'][0]['center']
        normal = Op['faces'][0]['normal']
        

        if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            normal[0] = 1

        self.prev_sketch = Preprocessing.proc_CAD.build123.protocol.build_circle(self.Op_idx, radius, center, normal, self.output, self.data_directory)
        self.Op_idx += 1
        self.circle_center = center


    def parse_extrude(self, Op, sketch_Op):

        sketch_point_list = [vert['coordinates'] for vert in sketch_Op['vertices']]
        normal_vec = self.prev_sketch.faces()[0].normal_at()
        sketch_face_normal = np.array([normal_vec.X, normal_vec.Y, normal_vec.Z])
        extrude_amount = Op['operation'][2]
        extrude_direction = np.array(Op['operation'][3])

        if np.allclose(-sketch_face_normal, extrude_direction, atol=1e-1):
            extrude_amount = -extrude_amount

        
        if self.canvas is None:
            self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)
        else:
            if random.random() < 0.4:
                self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)
            else:
                self.canvas = Preprocessing.proc_CAD.build123.protocol.build_subtract(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)

        self.Op_idx += 1
        

    def parse_fillet(self, Op):
        
        
        target_output_edge = Op['operation'][1]
        if len(Op['operation']) > 5:
            target_output_edge = Op['operation'][-1]
            
        target_edge, fillet_amount = Preprocessing.proc_CAD.helper.get_fillet_amount(target_output_edge, self.canvas.edges())


        if target_edge != None:
            self.canvas = Preprocessing.proc_CAD.build123.protocol.build_fillet(self.Op_idx, self.canvas, target_edge, fillet_amount * 0.8, self.output, self.data_directory)
            self.Op_idx += 1
        else:
            print("no target fillet edge")
            
    def parse_chamfer(self, Op):
        chamfer_amount = Op['operation'][2]['amount']
        verts = Op['operation'][3]['old_verts_pos']

        target_edge = Preprocessing.proc_CAD.helper.find_target_verts(verts, self.canvas.edges())

        if target_edge != None:
            self.canvas = Preprocessing.proc_CAD.build123.protocol.build_chamfer(self.Op_idx, self.canvas, target_edge, chamfer_amount, self.output, self.data_directory)
            self.Op_idx += 1

    def is_valid_parse(self):
        return self.Op_idx == self.len_program 


    def find_extrude_amount(self, sketch_point_list, sketch_face_normal, target_point):
        sketch_face_normal = np.array(sketch_face_normal)
        target_point = np.array(target_point)
        
        for sketch_point in sketch_point_list:
            sketch_point = np.array(sketch_point)
            vec = target_point - sketch_point
            extrude_amount = np.dot(vec, sketch_face_normal) / np.dot(sketch_face_normal, sketch_face_normal)
            
            # Check if reconstruction matches (with tolerance)
            reconstructed = sketch_point + extrude_amount * sketch_face_normal
            if np.allclose(reconstructed, target_point, atol=1e-4):
                return extrude_amount
        
        
        raise ValueError("Build123 : No matching sketch point found that can be extruded to target_point.")



# Example usage:

def run(data_directory = None):
    file_path = os.path.join(os.path.dirname(__file__), 'canvas', 'Program.json')
    if data_directory:
        file_path = os.path.join(data_directory, 'Program.json')

    parsed_program_class = parsed_program(file_path, data_directory)
    parsed_program_class.read_json_file()
    
    return parsed_program_class.is_valid_parse()

# run()