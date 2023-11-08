import torch
import torchvision
from torch import nn
import sys
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
from tqdm import tqdm
import json
import itertools

def iterate_dict(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def weight_init(model, init_type):
    if init_type == "orthogonal":
        torch.nn.init.orthogonal_(model.weight)
    elif init_type == "xavier":
        torch.nn.init.xavier_uniform_(model.weight)
    elif init_type == "kaiming":
        torch.nn.init.kaiming_uniform_(model.weight)
    else:
        pass

def fetch_data(cursor, batch_size, offset):
    cursor.execute(f'''
        SELECT file_name, file_content FROM file_attributes
        LIMIT {batch_size} OFFSET {offset}
    ''')
    return cursor.fetchall()

def save_model(model):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_models')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'model.pth'
    final_directory = os.path.join(directory, filename)
    torch.save(model.state_dict(), final_directory)

def yield_data_names(images, batch_size):
    start_index = 0
    stop_index = batch_size
    entities = len(images)

    while start_index < entities:
        batch_data = images[start_index:stop_index]
        start_index = stop_index
        stop_index += batch_size
        yield batch_data

def read_img_tgt(data_name_batch, new_path):
    images = []
    targets = []
    for file_name in data_name_batch:
        file_name += ".png"
        file_path = os.path.join(new_path, file_name)  
        img = cv2.imread(file_path)
        if img is None:
            continue
        if "up" in file_name[-8:]:
            targets.append(0)
        elif "down" in file_name[-8:]:
            targets.append(1)
        else:
            continue
        img = img.transpose(2, 0, 1)
        img = np.asarray(img)
        img = (img - np.min(img))/np.ptp(img)
        img = torch.from_numpy(img).double()
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        images.append(img)

    return torch.stack((images)), torch.Tensor(targets)

def load_model(model, model_name = 'model.pth'):
    print("Loading pretrained model ...")
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_models')
    model_name_path = os.path.join(saved_models_dir, model_name)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_name_path):
            model.load_state_dict(torch.load(model_name_path, map_location=torch.device('cpu')), strict=False)
        else:
            sys.exit("model path does not exist")
    else:
        sys.exit("path does not exist")

    return model

def hyperparam_optim(optimizer, 
                     loss_function, 
                     train_filenames, 
                     validation_file_names, 
                     max_epochs, 
                     model, 
                     params_dict, 
                     save_dir, 
                     best_loss,
                     batch_size,
                     device, 
                     data_path):
    
    consecutive_no_improvement = 0
    best_model_state = None
    best_params = None

    learning_rate = params_dict.get('learning_rate')
    overall_best_loss = best_loss

    for epoch in tqdm(range(max_epochs)):

        model.train()

        for data_name_batch in yield_data_names(train_filenames, batch_size):
            images, targets = read_img_tgt(data_name_batch, data_path)
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model.forward(images)
            loss = loss_function(pred, targets)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():

            val_loss = 0

            for data_name_batch in yield_data_names(validation_file_names, batch_size):
                images, targets = read_img_tgt(data_name_batch, data_path)
                images, targets = images.to(device), targets.to(device)
                pred = model.forward(images)
                val_loss += loss_function(pred, targets).item()

        val_loss = val_loss / len(validation_file_names)

        if val_loss < overall_best_loss:
            overall_best_loss = val_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= 2:
                break
            
    # Save best model to disk
    torch.save(best_model_state, f"{save_dir}/best_model.pth")

    print("current best loss: ", best_loss)

    return overall_best_loss