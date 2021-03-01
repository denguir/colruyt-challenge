import os
import numpy as np
import pandas as pd
import torch
import utils
import torch.nn as nn
import json
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.ops.boxes import box_convert
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class ColruytDataset(Dataset):
    '''Colruyt dataset for food detection and classification'''
    def __init__(self, json_file, img_dir, img_size=None, transforms=None):
        '''
        - json_file: a .json file that has the COCO format
        - img_dir: path to the directory containing the images annotated in json_file
        - img_size: tuple or int - if specified, resizes images and bbox accordingly, else keep original sizes
        - transforms: set of transforms to apply on the images (i.e normalization, etc)
        '''
        super(ColruytDataset, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.frame_list = self._get_dataset(json_file)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = (img_size[0], img_size[1]) # hxw
        else:
            self.img_size = None # keep original size
    
    def _get_dataset(self, json_file):
        '''Build structured version of the dataset from a json_file'''
        with open(json_file) as jf:
            data = json.load(jf)
        self.images = pd.DataFrame(data['images']).rename(columns={'id': 'image_id'})
        self.annotations = pd.DataFrame(data['annotations']).rename(columns={'id': 'annotation_id'})
        self.categories = pd.DataFrame(data['categories']).rename(columns={'id': 'category_id'})
        df = self.images.merge(self.annotations, how='inner', on='image_id')\
                        .merge(self.categories, how='left', on='category_id')\
                        .sort_values(by='image_id')
        df['img_no'] = df.groupby('image_id').ngroup()
        return df
    
    def _get_img_data_by_idx(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.frame_list[self.frame_list['img_no'] == idx].sort_values(by='annotation_id')
        return data

    def get_height_and_width(self, idx):
        data = self._get_img_data_by_idx(idx)
        width = data.iloc[0]['width']
        height = data.iloc[0]['height']
        return height, width

    def __len__(self):
        # corresponds to len(images)
        return max(self.frame_list['img_no']) + 1
    
    def show_stats(self):
        fig, ax = plt.subplots()
        ax = self.frame_list['category_id'].value_counts().plot(kind='bar',
                            figsize=(14,8),
                            title="Number for each category")
        ax.set_xlabel("category id")
        ax.set_ylabel("count")
        fig.show()

    def __getitem__(self, idx):
        data = self._get_img_data_by_idx(idx)
        img_path = os.path.join(self.img_dir, data.iloc[0]['file_name'])
        img_id = data.iloc[0]['image_id']
        img = Image.open(img_path).convert("RGB")

        target = {'boxes': box_convert(torch.from_numpy(np.array(data['bbox'].values.tolist())),
                                        'xywh',
                                        'xyxy'),
                  'labels': torch.from_numpy(data['category_id'].values),
                  'image_id': torch.tensor([img_id]),
                  'area': torch.from_numpy(data['area'].values),
                  'iscrowd': torch.from_numpy(data['iscrowd'].values)
                }
        
        if self.img_size:
            h, w = data.iloc[0]['height'], data.iloc[0]['width']
            img = F.resize(img, self.img_size)
            # scaling x bbox
            target['boxes'][:, (0, 2)] = torch.round(target['boxes'][:, (0, 2)] * self.img_size[1] / w)
            # scaling y bbox
            target['boxes'][:, (1, 3)] = torch.round(target['boxes'][:, (1, 3)] * self.img_size[0] / h)
            # adjust area
            target['area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * (target['boxes'][:, 3] - target['boxes'][:, 1])

        if self.transforms:
            img = self.transforms(img)

        return img, target


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    
    dataset = ColruytDataset(json_file=f'../data/train/train_info.json',
                             img_dir=f'../data/train/images',
                             #img_size=(180, 240),
                             transforms=transforms.Compose([
                                       transforms.ToTensor(),])
                            )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    images, targets = next(iter(dataloader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]

    dataset.show_stats()

    cat_map = pd.read_csv('../data/categories.csv')
    for image, target in zip(images, targets):
        print(type(image))
        print(target)
        utils.visualize_dataset(image, target, cat_map)
