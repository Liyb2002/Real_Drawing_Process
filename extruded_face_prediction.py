import Preprocessing.dataloader
import Preprocessing.gnn_graph

import Encoders.gnn.gnn
import Encoders.helper

import Preprocessing.proc_CAD
import Preprocessing.proc_CAD.helper
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from Preprocessing.config import device
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

graph_encoder = Encoders.gnn.gnn.SemanticModule()
graph_decoder = Encoders.gnn.gnn.Extrude_Decoder()

extruded_face_encoder = Encoders.gnn.gnn.SemanticModule()
extruded_face_decoder = Encoders.gnn.gnn.Extruded_Face_Decoder()

graph_encoder.to(device)
graph_decoder.to(device)

extruded_face_encoder.to(device)
extruded_face_decoder.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(extruded_face_encoder.parameters()) + list(extruded_face_decoder.parameters()), lr=0.0004)
batch_size = 16
# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
extrude_prediction_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
extruded_face_save_dir = os.path.join(current_dir, 'checkpoints', 'extruded_face_prediction')
os.makedirs(extrude_prediction_dir, exist_ok=True)
os.makedirs(extruded_face_save_dir, exist_ok=True)

def load_extrude_strokes_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(extrude_prediction_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(extrude_prediction_dir, 'graph_decoder.pth')))

def load_models():
    extruded_face_encoder.load_state_dict(torch.load(os.path.join(extruded_face_save_dir, 'graph_encoder.pth')))
    extruded_face_decoder.load_state_dict(torch.load(os.path.join(extruded_face_save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(extruded_face_encoder.state_dict(), os.path.join(extruded_face_save_dir, 'graph_encoder.pth'))
    torch.save(extruded_face_decoder.state_dict(), os.path.join(extruded_face_save_dir, 'graph_decoder.pth'))




# ------------------------------------------------------------------------------# 

def compute_accuracy(valid_output, valid_batch_masks):
    batch_size = valid_output.shape[0] // 400
    correct = 0

    for i in range(batch_size):
        output_slice = valid_output[i * 400:(i + 1) * 400]
        mask_slice = valid_batch_masks[i * 400:(i + 1) * 400]

        max_output_value, max_output_index = torch.max(output_slice, dim=0)
        max_mask_value, max_mask_index = torch.max(mask_slice, dim=0)

        values_where_mask_is_1 = output_slice[mask_slice == 1]
        indices_where_mask_is_1 = torch.nonzero(mask_slice == 1).squeeze()

        # print(f"Graph {i}: Values where mask is 1: {values_where_mask_is_1.tolist()}, Indices: {indices_where_mask_is_1.tolist()}")
        # print(f"Graph {i}: max_output_index={max_output_index.item()}, max_mask_index={max_mask_index.item()}")

        if max_output_index == max_mask_index:
            correct += 1

    return correct

# ------------------------------------------------------------------------------# 



def train():
    print("DO EXTRUDED FACE PREDICTION")

    # Load the saved models
    load_extrude_strokes_models()  
    graph_encoder.eval()
    graph_decoder.eval()

    best_val_accuracy = 0


    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/cad2sketch_annotated')
    print(f"Total number of shape data: {len(dataset)}")
    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs (same as in training)
    for data in tqdm(dataset, desc=f"Building Graphs"):
        if data is None:
            continue

        # Extract the necessary elements from the dataset
        data_idx, program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'extrude':
            continue
        
        if loop_neighboring_vertical.shape[0] > 400:
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        sketch_operation_mask = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-2)
        sketch_stroke_idx = (sketch_operation_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(kth_operation, sketch_operation_mask, stroke_node_features)
        extrude_selection_mask = torch.tensor(extrude_selection_mask, dtype=torch.float)

        extrude_stroke_idx = (extrude_selection_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        

        if len(extrude_stroke_idx) == 0:
            continue

        # Find the sketch_loops
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in sketch_stroke_idx for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        sketch_loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)

        if not (extrude_selection_mask == 1).any() and not (sketch_loop_selection_mask == 1).any():
            continue


        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )
        gnn_graph.set_select_sketch(sketch_loop_selection_mask)

        x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        predicted_extrude_strokes = graph_decoder(x_dict)
        predicted_extrude_stroke_idx = (predicted_extrude_strokes > 0.5).nonzero(as_tuple=True)[0]
    

        gnn_graph.set_select_extrude_strokes(predicted_extrude_strokes)

        # this is the logic to find the extruded face loop
        extrude_face_loop_idx = (kth_operation == 2).nonzero(as_tuple=True)[0]
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            chosen_count = sum(1 for stroke in loop if stroke in extrude_face_loop_idx)
            if chosen_count == len(loop) :
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_chosen_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).flatten()

        # Find the indices where the value is 1
        ones_indices = (loop_chosen_mask == 1).nonzero(as_tuple=True)[0]
        
        if len(ones_indices) > 1:
            # Set all to 0
            loop_chosen_mask[ones_indices] = 0
            # Set only the first occurrence to 1
            loop_chosen_mask[ones_indices[0]] = 1

        # Reshape to (-1, 1) as in the original
        loop_selection_mask = loop_chosen_mask.reshape(-1, 1)


        gnn_graph.to_device_withPadding(device)
        loop_selection_mask = loop_selection_mask.to(device)
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)

        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), predicted_extrude_stroke_idx, data_idx)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_face_loop_idx, data_idx)


        
    print(f"Total number of preprocessed graphs: {len(graphs)}")
    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:], graphs[:]
    train_masks, val_masks = loop_selection_masks[:], loop_selection_masks[:]


    # Convert train and validation graphs to HeteroData
    hetero_train_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in train_graphs]
    padded_train_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in train_masks]

    hetero_val_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in val_graphs]
    padded_val_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in val_masks]

    # Create DataLoaders for training and validation graphs/masks
    graph_train_loader = DataLoader(hetero_train_graphs, batch_size=16, shuffle=False)
    mask_train_loader = DataLoader(padded_train_masks, batch_size=16, shuffle=False)

    graph_val_loader = DataLoader(hetero_val_graphs, batch_size=16, shuffle=False)
    mask_val_loader = DataLoader(padded_val_masks, batch_size=16, shuffle=False)


    epochs = 50
    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        extruded_face_encoder.train()
        extruded_face_decoder.train()
        train_correct = 0
        train_total = 0


        total_iterations = min(len(graph_train_loader), len(mask_train_loader))
        for hetero_batch, batch_masks in tqdm(zip(graph_train_loader, mask_train_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()

            x_dict = extruded_face_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = extruded_face_decoder(x_dict)

            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask
            loss = criterion(valid_output, valid_batch_masks)

            train_correct += compute_accuracy(valid_output, valid_batch_masks)
            train_total += valid_batch_masks.shape[0] / 400


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_loss /= len(train_graphs)
        val_loss = 0.0
        correct = 0
        total = 0

        extruded_face_encoder.eval()
        extruded_face_decoder.eval()
        with torch.no_grad():
            total_iterations_val = min(len(graph_val_loader), len(mask_val_loader))

            for hetero_batch, batch_masks in tqdm(zip(graph_val_loader, mask_val_loader), 
                                                  desc="Validation", 
                                                  dynamic_ncols=True, 
                                                  total=total_iterations_val):
                x_dict = extruded_face_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                output = extruded_face_decoder(x_dict)

                batch_masks = batch_masks.to(output.device).view(-1, 1)
                valid_mask = (batch_masks != -1).float()
                valid_output = output * valid_mask
                valid_batch_masks = batch_masks * valid_mask
                loss = criterion(valid_output, valid_batch_masks)

                correct += compute_accuracy(valid_output, valid_batch_masks)
                total += valid_batch_masks.shape[0] / 400

        
        val_loss /= len(val_graphs)
        

        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.7f} - Validation Loss: {val_loss:.7f} - Train Accuracy: {train_accuracy:.5f} - Validation Accuracy: {accuracy:.5f}")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            save_models()
            print(f"Models saved at epoch {epoch+1} with validation accuracy: {accuracy:.5f}")







def eval():
    # Load the saved models
    load_extrude_strokes_models()  
    load_models()
    graph_encoder.eval()
    graph_decoder.eval()

    extruded_face_encoder.eval()
    extruded_face_decoder.eval()

    best_val_accuracy = 0


    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/small')
    print(f"Total number of shape data: {len(dataset)}")
    graphs = []
    loop_selection_masks = []

    # Preprocess and build the graphs (same as in training)
    for data in tqdm(dataset, desc=f"Building Graphs"):
        if data is None:
            continue

        # Extract the necessary elements from the dataset
        data_idx, program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'extrude':
            continue
        
        if loop_neighboring_vertical.shape[0] > 400:
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)
        sketch_operation_mask = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-2)
        sketch_stroke_idx = (sketch_operation_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(kth_operation, sketch_operation_mask, stroke_node_features)
        extrude_selection_mask = torch.tensor(extrude_selection_mask, dtype=torch.float)
        extrude_stroke_idx = (extrude_selection_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
        
        remain_stroke_idx = ((kth_operation == 1) & (extrude_selection_mask == 0)).nonzero(as_tuple=True)[0]

        if len(extrude_stroke_idx) == 0:
            continue

        # Find the sketch_loops
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            if all(stroke in sketch_stroke_idx for stroke in loop):
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        sketch_loop_selection_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).reshape(-1, 1)

        if not (extrude_selection_mask == 1).any() and not (sketch_loop_selection_mask == 1).any():
            continue


        gnn_graph = Preprocessing.gnn_graph.SketchLoopGraph(
            stroke_cloud_loops, 
            stroke_node_features, 
            strokes_perpendicular, 
            loop_neighboring_vertical, 
            loop_neighboring_horizontal, 
            loop_neighboring_contained,
            stroke_to_loop,
            stroke_to_edge
        )
        gnn_graph.set_select_sketch(sketch_loop_selection_mask)

        x_dict = graph_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        predicted_extrude_strokes = graph_decoder(x_dict)
        predicted_extrude_stroke_idx = (predicted_extrude_strokes > 0.5).nonzero(as_tuple=True)[0]
    

        gnn_graph.set_select_extrude_strokes(predicted_extrude_strokes)

        # this is the logic to find the extruded face loop
        extruded_face_loop_idx = (kth_operation == 2).nonzero(as_tuple=True)[0]
        loop_chosen_mask = []
        for loop in stroke_cloud_loops:
            chosen_count = sum(1 for stroke in loop if stroke in extruded_face_loop_idx)
            if chosen_count == len(loop) :
                loop_chosen_mask.append(1)  # Loop is chosen
            else:
                loop_chosen_mask.append(0)  # Loop is not chosen
        
        loop_chosen_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).flatten()

        # Find the indices where the value is 1
        ones_indices = (loop_chosen_mask == 1).nonzero(as_tuple=True)[0]
        
        if len(ones_indices) > 1:
            # Set all to 0
            loop_chosen_mask[ones_indices] = 0
            # Set only the first occurrence to 1
            loop_chosen_mask[ones_indices[0]] = 1

        # Reshape to (-1, 1) as in the original
        loop_selection_mask = loop_chosen_mask.reshape(-1, 1)


        gnn_graph.to_device_withPadding(device)
        loop_selection_mask = loop_selection_mask.to(device)
        graphs.append(gnn_graph)
        loop_selection_masks.append(loop_selection_mask)


        x_dict = extruded_face_encoder(gnn_graph.x_dict, gnn_graph.edge_index_dict)
        predicted_extruded_face_loops = extruded_face_decoder(x_dict)
        predicted_extruded_face_loop_idx = torch.argmax(predicted_extruded_face_loops).item()
        predicted_extruded_face_stroke_idx = stroke_cloud_loops[predicted_extruded_face_loop_idx]
        
        # combined_idx = torch.cat([sketch_stroke_idx, remain_stroke_idx], dim=0)
        combined_idx = torch.cat([sketch_stroke_idx, torch.tensor(predicted_extruded_face_stroke_idx)])

        print("remain_stroke_idx", remain_stroke_idx)
        if ones_indices[0].item() != predicted_extruded_face_loop_idx:
            print("predicted_extruded_face_loop_idx", predicted_extruded_face_loop_idx)
            print("predicted_extruded_face_stroke_idx", predicted_extruded_face_stroke_idx)
            print('data_idx', data_idx)
            print("combined_idx", combined_idx)
        Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), [], data_idx)

        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(),  [], data_idx)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(),  [], data_idx)




#---------------------------------- Public Functions ----------------------------------#


eval()