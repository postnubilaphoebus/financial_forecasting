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
#images, image_targets = load_stock_images(new_path)
#train_data, val_data = images[640:], images[:640]
#train_labels, val_labels = image_targets[640:], image_targets[:640]

# Training loop
learning_rate = 1e-5
batch_size = 128
num_epochs = 10
validation_size = 1024
loss_fn = torch.nn.BCEWithLogitsLoss( reduction='mean')
model = five_day_cnn().double()
model = model.apply(five_day_cnn.init_weights)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
val_loss_list = []

conn = sqlite3.connect('file_attributes.db')
cursor = conn.cursor()
cursor.execute('SELECT file_name FROM file_attributes')
all_filenames = [row[0] for row in cursor.fetchall()]
total_data_size = len(all_filenames)
random.shuffle(all_filenames)

for epoch in tqdm(range(num_epochs)):

    temp_train_loss = []
    temp_val_loss = []

    # Shuffle the filenames before each epoch
    if validation_size < total_data_size:
        train_filenames = all_filenames[validation_size:]
        random.shuffle(train_filenames)
        validation_file_names = all_filenames[:validation_size]

    model.train()

    for data_name_batch in yield_data_names(train_filenames, batch_size):
        images, targets = read_img_tgt(data_name_batch, new_path)
        optim.zero_grad()
        pred = model.forward(images)
        loss = loss_fn(pred.squeeze(), targets)
        loss.backward()
        optim.step()
        temp_train_loss.append(loss.item())

    model.eval()

    for data_name_batch in yield_data_names(validation_file_names, batch_size):
        images, targets = read_img_tgt(data_name_batch, new_path)
        pred = model.forward(images)
        loss = loss_fn(pred.squeeze(), targets)
        temp_val_loss.append(loss.item())

    train_loss_list.append(sum(temp_train_loss) / len(temp_train_loss))
    val_loss_list.append(sum(temp_val_loss) / len(temp_val_loss))

    print("val loss epoch", val_loss_list[-1])
    print("train loss epoch", train_loss_list[-1])

    if epoch > 1:
        last_three_epochs = sum(val_loss_list[-2:]) / 2
        three_epochs_ago = val_loss_list[-3]
        if last_three_epochs > three_epochs_ago:
            save_model(model)
            print("train_loss_list", train_loss_list)
            print("val_loss_list", val_loss_list)
            break

# Specify the file path where you want to save the list
file_path = "loss.txt"

# Open the file in write mode
with open(file_path, 'w') as file:
    for train, val in zip(train_loss_list, val_loss_list):
        file.write(f"{train}, {val}\n")

conn.close()