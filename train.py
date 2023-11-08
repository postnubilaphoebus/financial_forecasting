import torch
import torchvision
from torch import nn
import sys
import cv2
import numpy as np
from cnn import five_day_cnn
import os
from os import listdir
from os.path import isfile, join
import random
from tqdm import tqdm
import sqlite3
import math
from utils import weight_init, hyperparam_optim, iterate_dict
import json

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

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Data loading
current_path = os.getcwd()
new_path = os.path.join(current_path, "data")

# Training loop
learning_rate = 1e-5
batch_size = 128
num_epochs = 5
validation_size = 10240

parameters_dict = {"learning_rate": learning_rate, 
                   "batch_size": batch_size, 
                   "num_epochs": num_epochs, 
                   "validation_size": validation_size}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
train_loss_list = []
val_loss_list = []

conn = sqlite3.connect('file_attributes.db')
cursor = conn.cursor()
cursor.execute('SELECT file_name FROM file_attributes')
all_filenames = [row[0] for row in cursor.fetchall()]
total_data_size = len(all_filenames)
current_directory = os.getcwd()
save_model_dir = os.path.join(current_directory, r'saved_models')
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

########### Hyperparameter dictionary ############
hyperparameters_dict = {"learning_rate": [1e-4, 1e-5, 1e-6],
                        "dropout": [0.4, 0.5, 0.6],
                        "weight_init_type": ["orthogonal", "xavier", "kaiming"]}
##################################################

gridsearchtype = "exhaustive" #or "random"
max_iters = math.prod([len(sub) for sub in list(hyperparameters_dict.values())])
num_iters = 10

if gridsearchtype == "random" and num_iters < max_iters:
    best_loss = float('inf')
    for _ in tqdm(range(num_iters)):

        if validation_size < total_data_size:
            train_filenames = all_filenames[validation_size:]
            random.shuffle(train_filenames)
            validation_file_names = all_filenames[:validation_size]

        lr = random.sample(hyperparameters_dict.get("learning_rate"))
        dropout = random.sample(hyperparameters_dict.get("dropout"))
        weight_init_type = random.sample(hyperparameters_dict.get("weight_init_type"))
        model = five_day_cnn(dropout).double().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if weight_init_type ==  "orthogonal":
            model = model.apply(five_day_cnn.orthogonal_init)
        elif weight_init_type ==  "xavier":
            model = model.apply(five_day_cnn.xavier_init)
        elif weight_init_type ==  "kaiming":
            model = model.apply(five_day_cnn.kaiming_init)
        else:
            pass

        parameters_dict = {"learning_rate": lr}
        best_loss = hyperparam_optim(optim, 
                                     loss_fn, 
                                     train_filenames, 
                                     validation_file_names, 
                                     num_epochs, 
                                     model, 
                                     parameters_dict, 
                                     save_model_dir, 
                                     best_loss,
                                     batch_size,
                                     device,
                                     new_path)
        
        params_list = [lr, dropout, weight_init_type, best_loss]
        
        with open('hyperparams.txt', 'a') as f:
            json.dump(params_list, f)

elif gridsearchtype == "exhaustive":
    best_loss = float('inf')

    for params_dict in iterate_dict(hyperparameters_dict):

        if validation_size < total_data_size:
            train_filenames = all_filenames[validation_size:]
            random.shuffle(train_filenames)
            validation_file_names = all_filenames[:validation_size]

        lr = params_dict.get("learning_rate")
        dropout = params_dict.get("dropout")
        weight_init_type = params_dict.get("weight_init_type")

        model = five_day_cnn(dropout).double().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if weight_init_type ==  "orthogonal":
            model = model.apply(five_day_cnn.orthogonal_init)
        elif weight_init_type ==  "xavier":
            model = model.apply(five_day_cnn.xavier_init)
        elif weight_init_type ==  "kaiming":
            model = model.apply(five_day_cnn.kaiming_init)
        else:
            pass
        parameters_dict = {"learning_rate": lr}
        best_loss = hyperparam_optim(optim, 
                                     loss_fn, 
                                     train_filenames, 
                                     validation_file_names, 
                                     num_epochs,
                                     model, 
                                     parameters_dict, 
                                     save_model_dir, 
                                     best_loss,
                                     batch_size,
                                     device,
                                     new_path)
        
        params_list = [lr, dropout, weight_init_type, best_loss]
        
        with open('hyperparams.txt', 'a') as f:
            json.dump(params_list, f)
else:
    pass

used_validation_names = "used_validation_names.txt"

with open(used_validation_names, "w") as file:
    for name in validation_file_names:
        file.write(f"{name}\n")

conn.close()