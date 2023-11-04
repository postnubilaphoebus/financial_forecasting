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
            model.load_state_dict(torch.load(model_name_path), strict = False)
        else:
            sys.exit("model path does not exist")
    else:
        sys.exit("path does not exist")

    return model

current_path = os.getcwd()
new_path = os.path.join(current_path, "data")
images, image_targets = load_stock_images(new_path)
train_data, val_data = images[640:], images[:640]
train_labels, val_labels = image_targets[640:], image_targets[:640]

model = five_day_cnn().double()
model = load_model(model)

# down = 1, up = 0
hits = []

for minibatch, targets in yield_data(val_data, val_labels, 128):
    pred = model.forward(minibatch)
    pred = torch.sigmoid(pred.squeeze()).tolist()
    targets = targets.tolist()
    pred = [1 if x > 0.5 else 0 for x in pred]

    for p, l in zip(pred, targets):
        if p == l:
            hits.append(1)
        else:
            hits.append(0)

print("accuracy", sum(hits) / len(hits))



