# baseline with ControlNeXt 
### Dataset Setting
1. Download BDD100k dataset from the following url : https://dl.cv.ethz.ch/bdd100k/data/ 

    We need to download: **[10k_images_train.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip), [10k_images_val.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip), [10k_images_test.zip](https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip), and [bdd100k_sem_seg_labels_trainval.zip](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip)** 

2. Download json file: 

3. unzip each zip files and put them:
    ```bash
    ├── bdd100k
    │   ├── bdd100k
    │   ├──── images
    │   ├────── 10k 
    │   ├──────── test
    │   ├──────── train 
    │   └──────── val  
    │   ├──── labels
    │   ├────── sem_seg
    │   ├──────── colormaps
    │   ├──────── masks
    │   ├──────── polygons
    │   ├──────── rles
    │   └──────── customized_sem_seg_train.json

    ``` 