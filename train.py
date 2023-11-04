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

def save_model(model):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_models')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'model.pth'
    final_directory = os.path.join(directory, filename)
    torch.save(model.state_dict(), final_directory)

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

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Data loading
current_path = os.getcwd()
new_path = os.path.join(current_path, "data")
images, image_targets = load_stock_images(new_path)
train_data, val_data = images[640:], images[:640]
train_labels, val_labels = image_targets[640:], image_targets[:640]

# Training loop
learning_rate = 1e-5
batch_size = 128
loss_fn = torch.nn.BCEWithLogitsLoss( reduction='sum')
model = five_day_cnn().double()
model = model.apply(five_day_cnn.init_weights)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
val_loss_list = []

for epoch in tqdm(range(100)):

    temp_train_loss = []
    temp_val_loss = []

    model.train()

    for minibatch, targets in yield_data(train_data, train_labels, batch_size):
        optim.zero_grad()
        pred = model.forward(minibatch)
        loss = loss_fn(pred.squeeze(), targets)
        loss.backward()
        optim.step()
        temp_train_loss.append(loss.item())

    model.eval()
    
    for minibatch, targets in yield_data(val_data, val_labels, batch_size):
        pred = model.forward(minibatch)
        loss = loss_fn(pred.squeeze(), targets)
        temp_val_loss.append(loss.item())

    train_loss_list.append(sum(temp_train_loss) / len(temp_train_loss))
    val_loss_list.append(sum(temp_val_loss) / len(temp_val_loss))

    print("val loss epoch", val_loss_list[-1])
    print("train loss epoch", train_loss_list[-1])

    if epoch > 2:
        last_three_epochs = sum(val_loss_list[-3:]) / 3
        four_epochs_ago = val_loss_list[-4]
        if last_three_epochs > four_epochs_ago:
            save_model(model)
            print("train_loss_list", train_loss_list)
            print("val_loss_list", val_loss_list)
            break










    
