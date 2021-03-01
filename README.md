# Colruyt products detection
This code trains a [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf) architecture with a Resnet 50 backbone to learn to detect and categorize a set of 60 types of Colruyt products (plus the background). The code is written in Pytorch.

## How to use

### Install environment
To install the conda environment on which I developed the solution:
```
conda env create -f environment.yml
```
Then activate it:
```
conda activate gym_env
```

### Prepare the data
This repository contains no images since it is too heavy to upload, however you can build the expected folder structure by following these steps:
1. under __data__ folder, you will find an empty __images__ folder, populate it with the images of the exercise, there should be exactly 31.000 images
2. under __data__ folder, launch ```python3 separate_data.py```, this script will move the images to 4 different folders:
    * __train/images__
    * __test/images__
    * __val/images__
    * __unlabeled/images__
3. That's it you are ready

### Generate testing results
The results should be already available under __/data/test/result.csv__ in the format asked by Colruyt team.
To re-generate the expected csv results file, launch __export_test.py__ script, you need to configure the following parameter:
* model_path: path to your trained model, if it does not exists, a pretrained COCO model is loaded (with poor results)

### Training
To train the model, launch the __training.py__ script, you need to configure the following parameters:
* BATCH_SIZE: this value depends on the GPU memory available (default: 4)
* EPOCHS: how long do you want to train (default: 200)
* checkpoint_model: starts the training by loading the weights of this model (if it does not exists, a pretrained COCO model is loaded)
* new_model: path where to store new model

__NOTE__:
The training set has been extended by manually labeling a set of 300 extra images that were unlabeled and that did not belong to the test set. This has been done using the following repository:
https://github.com/jsbroks/coco-annotator


### Evaluation on validation set
A custom validation set of 100 difficult images has been labeled manually using the [COCO-annotator repo](https://github.com/jsbroks/coco-annotator). This validation does not contain images from the test set.
You can assess your model by following this steps:
* Under __src__, launch ```git clone https://github.com/philferriere/cocoapi.git```
* And install pycocotools with ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```
* In __prepare_eval.py__ script, select your model under __model_name__ variable
* Launch ```python3 prepare_eval.py```
* A result is generated in the folder __data/val/results__
* Go to __src/cocoapi/PythonAPI/demos__ and move __evaluation.ipynb__ notebook to this directory
* Configure the variables __annDir__, __annFile__ and __resFile__ according to your machine and your model name
* lauch __evaluation.ipynb__ notebook


### Visualize
To visualize the model on the validation set, launch __visualize.py__, you need to configure the following parameter:
* model_path: path to your trained model, if it does not exists, a pretrained COCO model is loaded (with poor results)
* Launch ```python3 training.py```
* Press q to go to the next image 

## Expected folder structure
.  
├── data  
│   ├── images  
│   ├── info.docx  
│   ├── separate_data.py  
│   ├── test  
│   │   ├── images  
│   │   ├── result.csv  
│   │   └── test.json  
│   ├── train  
│   │   ├── images  
│   │   ├── train_info.json  
│   │   ├── train.json  
│   │   └── train_results.json  
│   ├── unlabeled  
│   │   ├── convert_unlabeled.py  
│   │   ├── images  
│   │   ├── train_info_ext.json  
│   │   └── val_info_ext.json  
│   └── val  
│       ├── images  
│       ├── results  
│       ├── val_info.json  
│       └── val.json  
├── environment.yml  
├── model  
│   ├── fasterrcnn_resnet50_fpn_bb5_p3.pt  
├── README.md  
├── runs  
└── src  
    ├── cocoapi  
    ├── ColruytDataset.py  
    ├── export_test.py  
    ├── prepare_eval.py  
    ├── training.py  
    ├── utils.py  
    └── visualize.py  


