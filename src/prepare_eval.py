import torch
import copy
import utils
import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from einops import rearrange
from torchvision.ops import nms
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_image_id(fname):
    file_id = int(fname.split('.')[0].split('_')[1])
    return file_id


if __name__ == '__main__':
    
    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Faster RCNN pretrained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 61 # background + 60 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Loading saved model
    model_name = 'fasterrcnn_resnet50_fpn_bb5_p3'
    model_path =  f'../model/{model_name}.pt'
    utils.load_model(model, model_path)
    model.eval()

    # normalization
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    # load evaluations set
    eval_path = '../data/val'
    eval_file = 'val.json'
    result_file = f'results/val_results_{model_name}.json'
    with open(os.path.join(eval_path, eval_file)) as f:
        eval_images = [img_name.strip() for img_name in f.readlines()[1:]]
    
    results = []
    # Apply model on evaluation set
    with torch.no_grad():
        for img_name in tqdm(eval_images):
            img_path = os.path.join(eval_path, 'images', img_name)
            img_cv = cv2.imread(img_path)
            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32)
            # img = cv2.resize(img, dsize=(240, 180), interpolation=cv2.INTER_CUBIC)
            img = rearrange(torch.from_numpy(copy.deepcopy(img)), 'h w c -> c h w').to(device)
            img = norm(img)
            detections = model([img])
            detections = {k: v.cpu() for k, v in detections[0].items()}
            keep = nms(detections['boxes'], detections['scores'], 0.8)
            detections = {k: v[keep].cpu() for k, v in detections.items()}

            for i in range(len(detections['labels'])):
                xmin = detections['boxes'][i][0].item()
                ymin = detections['boxes'][i][1].item()
                width = detections['boxes'][i][2].item() - xmin
                height = detections['boxes'][i][3].item() - ymin
                det = {'bbox': [xmin, ymin, width, height],
                        'score': detections['scores'][i].item(),
                        'category_id': detections['labels'][i].item(),
                        'image_id': get_image_id(img_name)}
                results.append(det)

    with open(os.path.join(eval_path, result_file), 'w') as f:
        json.dump(results, f)

            

