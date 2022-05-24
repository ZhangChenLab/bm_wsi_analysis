## Highly accurate automated diagnosis of malignant hematological diseases based on whole-slide image bone marrow analysis using deep learning


## Main requirements

  * Linux (Tested on Ubuntu 18.04)
  * NVIDIA GPU
  * Python 3, Detectron2, h5py, matplotlib, numpy, opencv, openslide, pandas, pillow, PyTorch (1.7), scikit-learn, scipy, labelme



## Trail Extraction agent

### get patches
``` 
python ./trail_extraction/get_patches.py --wsi_route DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch --seg_filter
```
The above command will automatically process the bone marrow WSI into patches.

### get images
``` 
python ./trail_extraction/get_imgs.py --coords_route LABEL_DIRECTORY --wsi_route DATA_DIRECTORY --num_process 8 --save_root SAVE_PATH
```
The above command will automatically process the bone marrow WSI into tiled imgs.


### train

``` 
python ./trail_extraction/train.py --txt_path DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --task SAVE_NAME
```


## Cell Recognition agent

### get single-cell images

Please ensure that Detectron2 is installed correctly, more detailed help can be found at https://github.com/facebookresearch/detectron2.
``` 
python ./cell_recognition/get_single_cell.py --data_path DATA_DIRECTORY --save_route RESULTS_DIRECTORY 
```


### bone marrow classification

``` 
python ./cell_recognition/train.py --data_path DATA_DIRECTORY --save_route RESULTS_DIRECTORY --task SAVE_NAME
```

### References
Appreciate the great work from the following repositories:

- [paper](https://www.nature.com/articles/s41551-020-00682-w)
- [Detectron2](https://github.com/facebookresearch/detectron2)

### Citation
