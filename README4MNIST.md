# How to train/evaluate on MNIST dataset

The goal is to train on MNIST to go from 16x16x1 to 28x28x1 images.

# Download MNIST 28x28

Download the mnist train dataset in: dataset/mnist
Download the mnist val dataset in: dataset/mnist_val

# Resize to get 14×14 LR_IMGS and 28×28 HR_IMGS, then prepare 28×28 Fake SR_IMGS by bicubic interpolation

Rescale the image from 28x28 to 14x14 using bicubic interpolation and then rescale it back to 28x28 using bicubic interpolation. Also define the folder structure (classic or LMDB for faster loading) and the image format (RGB or greyscale). The images are cropped into squares from the center of the image.

```bash
python data/prepare_data.py  --path dataset/mnist  --out dataset/mnist --size 14,28 -l --n_worker 12 --resample bicubic --scale L

python data/prepare_data.py  --path dataset/mnist_val  --out dataset/mnist_val --size 14,28 -l --n_worker 12 --resample bicubic --scale L
```

 the parameters for the command above are:
    - path: the path to the .png image dataset
    - out: the path prefix to save the prepared dataset
    - size: the size of the LR and HR images
    - n_worker: the number of cpus to use for the preparation
    - resample: the resampling method to use for upscaling the images
    - lmdb (-l): whether to save the dataset in lmdb format for faster speed
    - scale: the color scale of the images, either 'L' for grayscale or 'RGB' for color

This will create the following folder structure used for training:

By not using LMDB (without --lmdb) the dataset folder will use a simple folder structure:
``` dataset/mnist_14_28/lr_14/
    dataset/mnist_14_28/hr_28/
    dataset/mnist_14_28/sr_28/
```
By using LMDB (recommended --lmdb) the dataset folder will use a LMDB structure:
``` dataset/mnist_14_28/.mdb
```

Note: By default the images are loaded in grayscale using -- scale L. If you want to load them in RGB use --scale RGB

## Modify the config file before training

- All the training, evaluation and inference paramaters are stored in: config/sr_sr3_16_28.json

    "datasets": {
        "train": {
            "name": "mnist",
            "mode": "HR", // whether need LR img
            "dataroot": "dataset/mnist_14_28",
            "datatype": "lmdb", //lmdb or img, path of img files
            "l_resolution": 14, // low resolution need to super_resolution
            "r_resolution": 28, // high resolution
            "batch_size": 16,
            "num_workers": 8,
            "use_shuffle": true,
            "data_len": -1 // -1 represents all data used in train
            "scale": "L" // L for grayscale, RGB for color
        },
        "val": {
            "name": "mnist_val",
            "mode": "LRHR",
            "dataroot": "dataset/mnist_val_14_28",
            "datatype": "lmdb", //lmdb or img, path of img files
            "l_resolution": 14,
            "r_resolution": 28,
            "data_len": 50,     
            "scale": "L" // L for grayscale, RGB for color
        }
    },

# data pipeline 

First the function create_dataset() located in data/__init__.py is called to instanciate the LRHRDataset class. The dataset is created using the dataroot, datatype, l_resolution, r_resolution, mode, data_len, use_shuffle, batch_size and num_workers parameters. 

### parameters of create_dataset:
    •   dataroot: Path to the dataset (can be a folder for images or an LMDB database).
	•	datatype: Specifies 'lmdb' for LMDB format or 'img' for image files.
	•	l_resolution and r_resolution: Low and high resolutions for the dataset.
	•	split: Either 'train' or 'val', indicating the dataset split.
	•	data_len: Optional limit on the dataset length this is useful for debugging or evaluation.
	•	need_LR: Whether to include low-resolution images in the output (only for mode 'LRHR').

## Training 

then you need to change the dataset config to your data path and image resolution in the config file. 

```json

"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/mnist_14_28", // path of dataset
        "l_resolution": 14, // low resolution need to super_resolution
        "r_resolution": 28, // high resolution
        "datatype": "lmbd", //lmdb or img, path of img files
    }
},
```
Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
Edit json files to adjust network structure and hyperparameters
```bash
python sr.py --phase train --config config/sr_sr3_14_28.json -enable_wandb -log_wandb_ckpt -debug
```





