import torch
import torchvision
from torch import nn
import sys
import cv2
import numpy as np
from cnn import five_day_cnn
import os
import random
from tqdm import tqdm

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

def yield_data(images, targets, batch_size):
    start_index = 0
    stop_index = batch_size
    entities = images.shape[0]

    while start_index < entities:
        batch_data = images[start_index:stop_index]
        batch_targets = targets[start_index:stop_index]
        start_index = stop_index
        stop_index += batch_size
        yield batch_data, batch_targets

def load_stock_images(path):
    images = []
    targets = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is None:
            continue
        if "up" in filename[-8:]:
            targets.append(0)
        elif "down" in filename[-8:]:
            targets.append(1)
        else:
            continue
        img = img.transpose(2, 0, 1)
        img = np.asarray(img)
        img = (img - np.min(img))/np.ptp(img)
        img = torch.from_numpy(img).double()
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        images.append(img)

    # shuffling still missing
    return torch.stack(images), torch.FloatTensor(targets)


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

current_path = os.getcwd()
data_path = os.path.join(current_path, "data")
val_name_path = os.path.join(current_path, "saved_models")
validation_files = os.path.join(val_name_path, "used_validation_names.txt")
validation_file_names = []

with open(validation_files, "r") as file:
    for line in file:
        validation_file_names.append(line.strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
model = five_day_cnn().double()
model = load_model(model).to(device)

# down = 1, up = 0
hits = []

for data_name_batch in tqdm(yield_data_names(validation_file_names, batch_size)):
    images, targets = read_img_tgt(data_name_batch, data_path)
    images, targets = images.to(device), targets.to(device)
    pred = model.forward(images)
    pred = [1 if x > 0.5 else 0 for x in pred]

    for p, l in zip(pred, targets):
        if p == l:
            hits.append(1)
        else:
            hits.append(0)

print("accuracy", sum(hits) / len(hits))