# How to train/evaluate on MNIST dataset

The goal is to train on MNIST to go from 16x16x1 to 28x28x1 images.

# Download MNIST 28x28

Download the mnist train dataset in: dataset/mnist
Download the mnist val dataset in: dataset/mnist_val

# Resize to get 14×14 LR_IMGS and 28×28 HR_IMGS, then prepare 28×28 Fake SR_IMGS by bicubic interpolation

Rescale the image from 28x28 to 14x14 using bicubic interpolation and then rescale it back to 28x28 using bicubic interpolation. Also define the folder structure (classic or LMDB for faster loading) and the image format (RGB or greyscale). The images are cropped into squares from the center of the image.

```bash
python data/prepare_data.py  --path dataset/mnist  --out dataset/mnist_14_28 --size 14,28 -l --n_worker 8 --resample bicubic --scale L
```
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

## Modify the config file

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
        },
        "val": {
            "name": "mnist_val",
            "mode": "LRHR",
            "dataroot": "dataset/mnist_val_14_28",
            "datatype": "lmdb", //lmdb or img, path of img files
            "l_resolution": 14,
            "r_resolution": 28,
            "data_len": 50
        }
    },
