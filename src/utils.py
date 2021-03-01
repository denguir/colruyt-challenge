import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from einops import rearrange

def load_model(model, model_path):
    model_name = model_path.split('/')[-1]
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loading of model {model_name} succesful.')
    except FileNotFoundError as e:
        print(e)
        print('No checkpoint available.')
        print(f'Initilialisation of random weights for {model_name}.')


def load_optimizer(optimizer, model_path):
    model_name = model_path.split('/')[-1]
    try:
        checkpoint = torch.load(model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loading of optimizer {model_name} succesful.')
    except FileNotFoundError as e:
        print(e)
        print('No checkpoint available.')
        print(f'Initilialisation of a new optimizer for {model_name}.')


def load_loss(model_path):
    model_name = model_path.split('/')[-1]
    try:
        checkpoint = torch.load(model_path)
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']
        print(f'Loading of {model_name} loss succesful.')
    except FileNotFoundError as fe:
        print(fe)
        print('No checkpoint available.')
        epoch, train_loss, val_loss = 0, np.inf, np.inf
    return epoch, train_loss, val_loss


def collate_fn(batch):
    # used for dataloader to read a batch correctly
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return [images, targets]


def visualize_dataset(image, target, cat_map=None):
    '''Visualize training set with labels.
    cat_map: dataframe of correspondance id->cat_name'''
    fig, ax = plt.subplots()
    ax.imshow(rearrange(image, 'c h w -> h w c'))
    boxes = target['boxes']
    labels = target['labels']
    for j in range(len(boxes)):
        xmin = boxes[j][0]
        ymin = boxes[j][1]
        width = boxes[j][2] - xmin
        height = boxes[j][3] - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_id = labels[j].item()
        if cat_map is not None:
            cat_name = cat_map[cat_map['category_id'] == label_id].name.item()
            super_cat_name = cat_map[cat_map['category_id'] == label_id].supercategory.item()
            ax.text(xmin, ymin - 10, f'{super_cat_name}: {cat_name}')
        else:
            ax.text(xmin, ymin - 10, f'{label_id}')
    plt.show()


def visualize_detection(image, detection, cat_map=None):
    '''Visualize detections with labels and scores
    cat_map: dataframe of correspondance id->cat_name'''
    fig, ax = plt.subplots()
    ax.imshow(rearrange(image, 'c h w -> h w c'))
    boxes = detection['boxes']
    labels = detection['labels']
    scores = detection['scores']
    for j in range(len(boxes)):
        xmin = boxes[j][0]
        ymin = boxes[j][1]
        width = boxes[j][2] - xmin
        height = boxes[j][3] - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_id = labels[j].item()
        score = round(scores[j].item(), 2)
        if cat_map is not None:
            cat_name = cat_map[cat_map['category_id'] == label_id].name.item()
            ax.text(xmin, ymin - 20, f'{cat_name}')
            ax.text(xmin, ymin, f'score: {score}')
        else:
            ax.text(xmin, ymin - 20 , f'id: {label_id}')
            ax.text(xmin, ymin, f'score: {score}')
    plt.show()

