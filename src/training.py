import torch
import sys
import utils
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ColruytDataset import ColruytDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 4 # this is the max for a GPU with 8GB of memory
EPOCHS = 200


if __name__ == '__main__':

    # Choosing device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Resnet50 backbone pretrained on COCO
    # trainable_backbone_layers = [0, 5], 0: Not trainable
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True,
                                                     trainable_backbone_layers=5)
    num_classes = 61 # background + 60 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Loading saved model from checkpoint
    # if the checkpoint_model is not found, it will start from the pretrained COCO model
    checkpoint_model =  f'../model/fasterrcnn_resnet50_fpn_bb5_p3.pt' 
    new_model =  f'../model/fasterrcnn_resnet50_fpn_bb5_p4.pt'
    utils.load_model(model, checkpoint_model)

    # Loading optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, # 0.005
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    utils.load_optimizer(optimizer, new_model)

    # Loading last losses and epoch
    last_epoch, last_train_loss, last_val_loss = utils.load_loss(new_model)

    # Loading dataset
    dataset = ColruytDataset(json_file=f'../data/train/train_info.json',
                             img_dir=f'../data/train/images',
                             # img_size=(180, 240),
                             transforms=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                       ])
                            )
    # Split dataset into training and validation sets
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, collate_fn=utils.collate_fn)

    # Training
    model.train()
    min_val_running_loss = last_val_loss
    writer = SummaryWriter(log_dir='../runs/bb5') # log to tensorboard
    for epoch in range(last_epoch + 1, last_epoch + 1 + EPOCHS):
        train_running_loss = 0.0
        val_running_loss = 0.0
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            # zero the parameters gradient
            optimizer.zero_grad()
            # load batch on device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # forward pass
            train_loss_dict = model(images, targets)
            train_loss = sum(loss for loss in train_loss_dict.values())
            # backward pass
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()

        with torch.no_grad():
            for j, (images, targets) in enumerate(tqdm(val_loader)):
                # load batch on device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # forward pass on validation set
                val_loss_dict = model(images, targets)
                val_loss = sum(loss for loss in val_loss_dict.values())
                val_running_loss += val_loss.item()

        # print stats
        train_running_loss /= len(train_loader)
        val_running_loss /= len(val_loader)
        print(f"Epoch {epoch}:\n\
                train_loss: {train_running_loss} \t val_loss: {val_running_loss}")

        # log stats
        writer.add_scalar('Loss/train', train_running_loss, epoch)
        writer.add_scalar('Loss/val', val_running_loss, epoch)

        if val_running_loss < min_val_running_loss:
            print(f"val_loss decreased: {min_val_running_loss} -> {val_running_loss}.")
            print(f"Saving model to {new_model}.")
            torch.save({
                'epoch': epoch,
                'train_loss': train_running_loss,
                'val_loss': val_running_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, new_model)
        min_val_running_loss = min(min_val_running_loss, val_running_loss)

