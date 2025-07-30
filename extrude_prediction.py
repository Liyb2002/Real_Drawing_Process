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

graph_encoder.to(device)
graph_decoder.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.0004)
batch_size = 16
# ------------------------------------------------------------------------------# 

current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'extrude_prediction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))

# ------------------------------------------------------------------------------# 



def compute_accuracy(valid_output, valid_batch_masks):
    batch_size = valid_output.shape[0] // 400
    correct = 0
    total_count = 3 * batch_size

    for i in range(batch_size):
        output_slice = valid_output[i * 400:(i + 1) * 400].squeeze()  # shape: [400]
        mask_slice = valid_batch_masks[i * 400:(i + 1) * 400].squeeze()  # shape: [400]

        top3_indices = torch.topk(output_slice, 3).indices

        for idx in top3_indices:
            pred = output_slice[idx]
            label = mask_slice[idx]

            if (pred > 0.5 and label == 1) or (pred < 0.5 and label == 0):
                correct += 1

    return correct, total_count



def compute_accuracy_eval(output, loop_selection_mask, hetero_batch, data_indices, padded_size=400):
    correct = 0
    total_loops = loop_selection_mask.shape[0] // padded_size  # Determine how many (400,1) matrices there are

    # Loop through each matrix of size (400, 1)
    for i in range(total_loops):
        data_idx = data_indices[i]
        start_idx = i * padded_size
        end_idx = start_idx + padded_size

        # Extract the (400, 1) slice for both output and loop_selection_mask
        output_slice = output[start_idx:end_idx]
        mask_slice = loop_selection_mask[start_idx:end_idx]
        stroke_node_features_slice = hetero_batch.x_dict['stroke'][i * 400:(i + 1) * 400]


        # Evaluate conditions for this slice
        condition_1 = (mask_slice == 1) & (output_slice > 0.5)
        condition_2 = (mask_slice == 0) & (output_slice < 0.5)

        # Print output values where loop_selection_mask == 1 for the current slice
        mask_1_indices = (mask_slice == 1).nonzero(as_tuple=True)
        # if mask_1_indices[0].numel() > 0:
        #     print(f"Output values where loop_selection_mask == 1 for slice {i}:")
        #     print(output_slice[mask_1_indices])

        # Check if all conditions are met for this slice
        if torch.all(condition_1 | condition_2):
            correct += 1
        else:
            # pass
            extrude_stroke_idx = (output_slice > 0.5).nonzero(as_tuple=True)[0]  # Indices of chosen strokes
            gt_stroke_idx = (mask_slice > 0.5).nonzero(as_tuple=True)[0]  # Indices of chosen strokes

            Encoders.helper.vis_selected_strokes(stroke_node_features_slice.cpu().numpy(), extrude_stroke_idx, data_idx)
            Encoders.helper.vis_selected_strokes(stroke_node_features_slice.cpu().numpy(), gt_stroke_idx, data_idx)


    return correct


# ------------------------------------------------------------------------------# 



def train():
    print("DO EXTRUDE PREDICTION")
    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/cad2sketch_annotated')
    print(f"Total number of shape data: {len(dataset)}")

    best_val_accuracy = 0

    graphs = []
    stroke_selection_masks = []

    # Preprocess and build the graphs
    for data in tqdm(dataset, desc=f"Building Graphs"):
        if data is None:
            continue

        # Extract the necessary elements from the dataset
        data_idx, program, program_whole, stroke_cloud_loops, stroke_node_features, strokes_perpendicular, output_brep_edges, stroke_operations_order_matrix, loop_neighboring_vertical, loop_neighboring_horizontal,loop_neighboring_contained, stroke_to_loop, stroke_to_edge = data

        if program[-1] != 'extrude'or len(program) > stroke_operations_order_matrix.shape[1]:
            continue
        
        if stroke_node_features.shape[0] > 400:
            continue

        kth_operation = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-1)

        if kth_operation is None:
            continue
        
        all_kth_stroke =  (kth_operation == 1).nonzero(as_tuple=True)[0]

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
        gnn_graph.to_device_withPadding(device)
        extrude_selection_mask = extrude_selection_mask.to(device)

        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

        # print("extrude_stroke_idx", extrude_stroke_idx)
        # print("gnn_graph['stroke'].x.cpu().numpy()", stroke_node_features.shape)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), sketch_stroke_idx, data_idx)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_stroke_idx, data_idx)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), all_kth_stroke, data_idx)


    print(f"Total number of preprocessed graphs: {len(graphs)}")

    # Split the dataset into training and validation sets (80-20 split)
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:], graphs[:]
    train_masks, val_masks = stroke_selection_masks[:], stroke_selection_masks[:]

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

    # Training and validation loop
    epochs = 30  # Number of epochs
    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        train_total = 0

        graph_encoder.train()
        graph_decoder.train()

        # Get total number of iterations for progress bar
        total_iterations = min(len(graph_train_loader), len(mask_train_loader))

        # Training loop with progress bar
        for hetero_batch, batch_masks in tqdm(zip(graph_train_loader, mask_train_loader), 
                                              desc=f"Epoch {epoch+1}/{epochs} - Training", 
                                              dynamic_ncols=True, 
                                              total=total_iterations):

            optimizer.zero_grad()
            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)
            output = graph_decoder(x_dict)

            # Ensure masks are on the correct device and reshape them to match the output
            batch_masks = batch_masks.to(output.device).view(-1, 1)
            valid_mask = (batch_masks != -1).float()  

            # Apply the valid mask to output and batch_masks
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask

            # Compute the loss only on valid (non-padded) values
            loss = criterion(valid_output, valid_batch_masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy computation using the preferred method (only on valid values)
            batch_correct, batch_total= compute_accuracy(valid_output, valid_batch_masks)
            correct += batch_correct
            train_total += batch_total

        train_accuracy = correct / train_total
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss / total_iterations:.5f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation loop
        val_loss = 0.0
        correct = 0
        total = 0
        graph_encoder.eval()
        graph_decoder.eval()

        with torch.no_grad():
            total_iterations_val = min(len(graph_val_loader), len(mask_val_loader))

            for hetero_batch, batch_masks in tqdm(zip(graph_val_loader, mask_val_loader), 
                                                  desc="Validation", 
                                                  dynamic_ncols=True, 
                                                  total=total_iterations_val):
                # Forward pass through the graph encoder
                x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)

                # Forward pass through the graph decoder
                output = graph_decoder(x_dict)

                # Ensure masks are on the correct device and reshape them
                batch_masks = batch_masks.to(output.device).view(-1, 1)

                # Apply the valid mask to output and batch_masks
                valid_mask = (batch_masks != -1).float()
                valid_output = output * valid_mask
                valid_batch_masks = batch_masks * valid_mask

                # Compute the validation loss
                loss = criterion(valid_output, valid_batch_masks)
                val_loss += loss.item()

                # Accuracy computation using the preferred method (only on valid values)
        
                batch_correct, batch_total= compute_accuracy(valid_output, valid_batch_masks)
                correct += batch_correct
                total += batch_total
        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / total_iterations_val:.5f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}, saved model")
            save_models()




def eval():
    # Load the saved models
    load_models()  
    graph_encoder.eval()
    graph_decoder.eval()

    batch_size = 16

    # Load the dataset
    dataset = Preprocessing.dataloader.Program_Graph_Dataset('dataset/small')
    print(f"Total number of shape data: {len(dataset)}")

    graphs = []
    data_indices = []
    stroke_selection_masks = []

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


        if kth_operation is None:
            continue

        sketch_operation_mask = Encoders.helper.get_kth_operation(stroke_operations_order_matrix, len(program)-2)
        sketch_stroke_idx = (sketch_operation_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen strokes

        
        extrude_selection_mask = Encoders.helper.choose_extrude_strokes(kth_operation, sketch_operation_mask, stroke_node_features)
        extrude_selection_mask = torch.tensor(extrude_selection_mask, dtype=torch.float)
        extrude_stroke_idx =  (extrude_selection_mask == 1).nonzero(as_tuple=True)[0]


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
        gnn_graph.to_device_withPadding(device)
        extrude_selection_mask = extrude_selection_mask.to(device)
        data_indices.append(data_idx)

        graphs.append(gnn_graph)
        stroke_selection_masks.append(extrude_selection_mask)

        if len(graphs) > 50:
            break
        
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), sketch_stroke_idx, data_idx)
        # Encoders.helper.vis_selected_strokes(gnn_graph['stroke'].x.cpu().numpy(), extrude_stroke_idx, data_idx)

        
    print(f"Total number of preprocessed graphs: {len(graphs)}")

    # Convert to HeteroData and pad the masks
    hetero_graphs = [Preprocessing.gnn_graph.convert_to_hetero_data(graph) for graph in graphs]
    padded_masks = [Preprocessing.dataloader.pad_masks(mask) for mask in stroke_selection_masks]

    # Create DataLoader for evaluation
    graph_eval_loader = DataLoader(hetero_graphs, batch_size=batch_size, shuffle=False)
    mask_eval_loader = DataLoader(padded_masks, batch_size=batch_size, shuffle=False)
    data_indices_loader = DataLoader(data_indices, batch_size=batch_size, shuffle=False)

    eval_loss = 0.0
    total = 0
    correct = 0

    criterion = torch.nn.BCELoss()  # Assuming BCELoss is used in the training

    with torch.no_grad():
        total_iterations_eval = min(len(graph_eval_loader), len(mask_eval_loader))

        for hetero_batch, batch_masks, data_indices in tqdm(zip(graph_eval_loader, mask_eval_loader, data_indices_loader), desc="Evaluation", dynamic_ncols=True, total=total_iterations_eval):
            # Forward pass through the graph encoder
            x_dict = graph_encoder(hetero_batch.x_dict, hetero_batch.edge_index_dict)

            # Forward pass through the graph decoder
            output = graph_decoder(x_dict)

            # Ensure masks are on the correct device and reshape them
            batch_masks = batch_masks.to(output.device).view(-1, 1)

            # Create a valid mask for non-zero values (padding values are -1)
            valid_mask = (batch_masks != -1).float()
            valid_output = output * valid_mask
            valid_batch_masks = batch_masks * valid_mask


            correct += compute_accuracy_eval(valid_output, valid_batch_masks, hetero_batch, data_indices)           
            total += batch_size

            # Compute loss
            loss = criterion(valid_output, valid_batch_masks)
            eval_loss += loss.item()


    # Calculate and print overall average accuracy
    overall_accuracy = correct / total
    print(f"Overall Average Accuracy: {overall_accuracy:.2f}%")


#---------------------------------- Public Functions ----------------------------------#


train()