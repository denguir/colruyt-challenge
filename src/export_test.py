import torch
import copy
import utils
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from einops import rearrange
from torchvision.ops import nms
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    model_path =  f'../model/fasterrcnn_resnet50_fpn_bb5_p3.pt'
    utils.load_model(model, model_path)
    model.eval()

    # Normalization
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    # Apply model on test set
    with open('../data/test/test.json') as f:
        test_images = [img_name.strip() for img_name in f.readlines()[1:]]
    
    # Generate results
    df = pd.DataFrame(columns=['image_name', 'id', 'confidence', 'x_center_scaled', 'y_center_scaled', 
                               'bbox_width', 'bbox_height'])
    print('Exporting results to csv ...')
    with torch.no_grad():
        for img_name in tqdm(test_images):
            img_path = f'../data/test/images/{img_name}'
            img_cv = cv2.imread(img_path)
            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32)
            # img = cv2.resize(img, dsize=(240, 180), interpolation=cv2.INTER_CUBIC)
            img = rearrange(torch.from_numpy(copy.deepcopy(img)), 'h w c -> c h w').to(device)
            img = norm(img)
            _, h, w = img.size()
            detections = model([img])
            detections = {k: v.cpu() for k, v in detections[0].items()}
            keep = nms(detections['boxes'], detections['scores'], 0.8)
            detections = {k: v[keep].cpu() for k, v in detections.items()}
            
            # populate result df
            img_name = img_name.split('.')[0]
            for i in range(len(detections['labels'])):
                x_min = detections['boxes'][i][0].item()
                y_min = detections['boxes'][i][1].item()
                x_max = detections['boxes'][i][2].item()
                y_max = detections['boxes'][i][3].item()
                bbox_width = (x_max - x_min) / w
                bbox_height = (y_max - y_min) / h
                xc = (x_min / w) + bbox_width / 2
                yc = (y_min / w) + bbox_height / 2 
                det = {'image_name': img_name,
                       'id': detections['labels'][i].item(),
                       'confidence': detections['scores'][i].item(),
                       'x_center_scaled': xc,
                       'y_center_scaled': yc,
                       'bbox_width': bbox_width,
                       'bbox_height': bbox_height}
                df = df.append(det, ignore_index=True)
    df.to_csv('../data/test/result.csv')
    print('Exportation succesful.')





            
  

