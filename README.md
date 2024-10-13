# baseline with ControlNeXt 
### Dataset Setting
1. Download BDD100k dataset from the following url : https://dl.cv.ethz.ch/bdd100k/data/ 

    We need to download: **[10k_images_train.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip), [10k_images_val.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip), [10k_images_test.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip), and [bdd100k_sem_seg_labels_trainval.zip](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip)** 

2. unzip each zip files and put them:
    ```bash
    ├── bdd100k
    │   ├── bdd100k
    │   ├──── images
    │   ├────── 10k 
    │   ├──────── test
    │   ├──────── train 
    │   └──────── val  
    │   ├──── labels
    │   ├────── 10k
    │   ├──────── sem_seg
    │   ├────────── colormaps
    │   ├────────── masks
    │   ├────────── polygons
    │   ├────────── rles
    │   ├────────── customized_sem_seg_train.json
    │   └────────── customized_sem_seg_val.json

    ``` 

3. Move the data inside ```customized_data/``` according to the above directory structure

### Conda environment 
1. Follow the environment setting of [AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) 

2. Then install diffusers library
    ```
    conda activate [가상환경이름]
    cd seg2real/diffusers
    pip install -e  .
    cd ..
    ```

### Quick start 
1. Modify configs/train/stage1.yaml file
    (1) data.real_images_path : the path to the directory of real images
    (2) data.seg_images_path : the path to the directory of segmentation masks
    (3) base_model_path : the path to the SD1.5 
    (4) vae_model_path : path to the vae of SD1.5 (`[base_model_path]/vae`)
    (5) controlnext_path : pretrained path of the ControlNeXt
    (6) pretrained_unet_path : pretrained path of the UNet 
    (7) output_dir : directory for saving checkpoints and validation results
    (8) validation.real_paths : list of the real reference images for validation
    (9) validation.seg_paths : list of the segmentation images for validation
    (10) validation.val_json : json file about object information for validation

2. for single gpu 
    ``` 
    CUDA_VISIBLE_DEVICES=$DEVICE python train_w_controlNeXt.py
    ```

3. for multi gpu
    ```
    bash train_multi.sh
    ```

    In the bash file, modify the device 