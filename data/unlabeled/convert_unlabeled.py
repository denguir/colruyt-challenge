import os
import sys
import json
import pandas as pd

def get_image_id(fname):
    file_id = int(fname.split('.')[0].split('_')[1])
    return file_id


def get_annot_id(img_id, annot_sub_id):
    annot_id = int(str(img_id) + "{0:0=3d}".format(annot_sub_id))
    return annot_id


def get_img_id_map(data):
    img_id_map = {}
    for img in data['images']:
        img_id_map[img['id']] = get_image_id(img['file_name'])
    return img_id_map


def convert_img_format(img_data, img_id_map):
    image_keys = ['file_name', 'height', 'width', 'id', 'license']
    for i in range(len(img_data)):
        img_data[i]['id'] = img_id_map[img_data[i]['id']]
        img_data[i]['license'] = 1
        img_data[i] = {k: img_data[i][k] for k in image_keys}
    return img_data


def convert_annot_format(annot_data, img_id_map):
    annot_keys = ['segmentation', 'area', 'iscrowd', 'image_id', 'category_id', 'bbox', 'id']
    for i in range(len(annot_data)):
        annot_data[i]['image_id'] = img_id_map[annot_data[i]['image_id']]
        annot_data[i]['id'] = get_annot_id(annot_data[i]['image_id'], annot_data[i]['id'])
        annot_data[i]['iscrowd'] = int(annot_data[i]['iscrowd'])
        annot_data[i] = {k: annot_data[i][k] for k in annot_keys}
    return annot_data


def convert_format(data):
    # treat images
    img_id_map = get_img_id_map(data)
    images = convert_img_format(data['images'], img_id_map)
    # treat annotations
    annotations = convert_annot_format(data['annotations'], img_id_map)
    return {'images': images, 'annotations': annotations}


if __name__ == '__main__':
    src_file = sys.argv[1] # train_info_ext.json or val_info_ext.json
    dest_folder = sys.argv[2] # train or val
    dest_file = dest_folder + '.json'
    dest_info_file = dest_folder + '_info.json'

    with open(src_file) as f:
        unlabeled_data = convert_format(json.load(f))

    with open(f'../{dest_folder}/{dest_info_file}') as f:
        data = json.load(f)
    
    # get new images
    img = pd.DataFrame(data['images'])
    unlabeled_img = pd.DataFrame(unlabeled_data['images'])
    common = unlabeled_img.merge(img, on=['id'])
    new_img = unlabeled_img[(~unlabeled_img.id.isin(common.id))]

    if len(new_img) > 0:
        print(f'Transfering {len(new_img)} new annotated images to {dest_folder} set ...')
        # get new annotations
        annots = pd.DataFrame(data['annotations'])
        unlabeled_annots = pd.DataFrame(unlabeled_data['annotations'])
        new_annots = unlabeled_annots[unlabeled_annots.image_id.isin(new_img.id)]

        # extend train_info.json
        data['images'] += list(new_img.T.to_dict().values())
        data['annotations'] += list(new_annots.T.to_dict().values())
        print('Transfering images:')
        print(new_img)
        with open(f'../{dest_folder}/{dest_info_file}', 'w') as f:
            extended_data = {'images': data['images'],
                                'annotations': data['annotations'],
                                'categories': data['categories']}
            json.dump(extended_data, f)
        
        with open(f'../{dest_folder}/{dest_file}', 'a') as f:
            for fname in new_img.file_name:
                # write to train.json or val.json
                f.write(fname + '\n')
                # move image to train/images or val/images
                os.rename(f'images/{fname}', f'../{dest_folder}/images/{fname}')
    else:
        print('No new annotated data to transfer.')