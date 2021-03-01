import os

# move train images to train folder
with open('train/train.json') as f:
    train_images = [img_name.strip() for img_name in f.readlines()[1:]]
    for img in train_images:
        if os.path.exists(f"images/{img}"):
            os.rename(f"images/{img}", f"train/images/{img}")

# move test images to test folder
with open('test/test.json') as f:
    test_images = [img_name.strip() for img_name in f.readlines()[1:]]
    for img in test_images:
        if os.path.exists(f"images/{img}"):
            os.rename(f"images/{img}", f"test/images/{img}")

# move val images to val folder
with open('val/val.json') as f:
    val_images = [img_name.strip() for img_name in f.readlines()[1:]]
    for img in val_images:
        if os.path.exists(f"images/{img}"):
            os.rename(f"images/{img}", f"val/images/{img}")
            
            
# move remaining images to unlabeled folder
for img in os.listdir('images'):
    if os.path.exists(f"images/{img}"):
        os.rename(f"images/{img}", f"unlabeled/images/{img}")
