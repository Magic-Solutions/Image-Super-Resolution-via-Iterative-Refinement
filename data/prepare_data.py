import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False, scale='L'):
    def add_channel_dimension(im):
        if scale == 'L' and isinstance(im, Image.Image):  # Only for grayscale
            im = np.expand_dims(np.array(im), axis=-1)  # Add channel dimension
            return Image.fromarray(np.squeeze(im, axis=-1))  # Convert back to Image for consistency
        return im

    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if scale == 'L':  # Add channel dimension for grayscale images
        lr_img = add_channel_dimension(lr_img)
        hr_img = add_channel_dimension(hr_img)
        sr_img = add_channel_dimension(sr_img)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False, scale='L'):  # scale = 'L' for grayscale, 'RGB' for color
    img = Image.open(img_file)
    img = img.convert(scale)
    out = resize_multiple(img, sizes=sizes, resample=resample, lmdb_save=lmdb_save, scale=scale)

    # Add channel dimension for grayscale images if not saving to LMDB
    if not lmdb_save and scale == 'L':
        out = [np.expand_dims(np.array(im), axis=-1) if isinstance(im, Image.Image) else im for im in out]

    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.sizes = sizes
        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    # Initialize lmdb environment inside the worker process if lmdb_save is True
    env = None
    if wctx.lmdb_save:
        env = lmdb.open(wctx.out_path, map_size=1024 ** 4, readahead=False)
    
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        else:
            with env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False, scale='RGB'):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save, scale=scale)
    files = [p for p in Path(img_path).glob('**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                                         sizes[0], sizes[1]), exist_ok=True)

    if n_worker > 1:
        # prepare data subsets
        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)

        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img = imgs
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
            else:
                env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,default='./dataset/celebahq')
    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')
    parser.add_argument('--scale', type=str, choices=['L', 'RGB'], default='RGB', 
                        help="Image color scale: 'L' for grayscale, 'RGB' for color")
    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb, scale = args.scale)
