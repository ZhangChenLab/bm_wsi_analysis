import sys
import os
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock
import openslide
import h5py
import argparse

count = Value('i', 0)
lock = Lock()

def process(opts):

    num_process = 6
    patch_size = 256
    level = 0
    i, pid, label,x, y,save_path = opts
    wsi_path = pid
    slide = openslide.OpenSlide(wsi_path)
    img = slide.read_region(
        (x, y), level,
        (patch_size, patch_size)).convert('RGB')
    patch_path = os.path.join(save_path, label)
    img.save(os.path.join(patch_path, pid + '_' + str(x)+'_'+str(y) + '.png'))
    global lock
    global count
    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            print('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))

def run(coords_path,wsi_route,num_process,tissue_route,mask_route,save_path,patient):
    opts_list = []

    pid = os.path.join(wsi_route,patient + '.tif')
    infile = h5py.File(coords_path)
    lines = infile['coords']
    for i in range(lines.shape[0]):
        point = infile['coords'][i]
        x = point[0]
        y = point[1]
        label = infile['labels'][i]
        if label == 0:
            new_label = 'tissue'
        else:
            new_label = 'mask'
        opts_list.append((i, pid,new_label ,x, y,save_path))
    infile.close()
    pool = Pool(processes=num_process)
    pool.map(process,opts_list)


def main(args):

    num_process = args.num_process
    coords_route = args.coords_route
    wsi_route = args.wsi_route
    patch_route = args.save_route
    if not os.path.exists(patch_route):
        os.makedirs(patch_route)
    patch_files = os.listdir(coords_route)
    for patch_file in patch_files:
        coords_path = os.path.join(os.path.join(coords_route,patch_file))
        patient = patch_file.split('.')[0]
        save_path = os.path.join(patch_route,patient)
        mask_route = os.path.join(patch_route,patient,'mask')
        tissue_route = os.path.join(patch_route,patient,'tissue')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(tissue_route):
            os.makedirs(tissue_route)
        if not os.path.exists(mask_route):
            os.makedirs(mask_route)
        run(coords_path,wsi_route,num_process,tissue_route,mask_route,save_path,patient)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get patch imgs')
    parser.add_argument('--coords_route', type=str, default='',
                        help='path to folder containing patch infos')
    parser.add_argument('--wsi_route', type=str, default='',
                        help='path to folder containing WSI')
    parser.add_argument('--num_process', type=int, default=6,
                        help='process')
    parser.add_argument('--save_route', type=str, default= '',
                        help='path to save patch imgs')

    args = parser.parse_args()
    main(args)