# MUSES

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-shot-temporal-event-localization-a/temporal-action-localization-on-thumos14)](https://paperswithcode.com/sota/temporal-action-localization-on-thumos14?p=multi-shot-temporal-event-localization-a)

This repo holds the code and the models for MUSES, introduced in the paper:<br>
[Multi-shot Temporal Event Localization: a Benchmark](https://arxiv.org/abs/2012.09434)<br>
[Xiaolong Liu](https://xlliu7.github.io), [Yao Hu](https://scholar.google.com/citations?user=LIu7k7wAAAAJ), [Song Bai](http://songbai.site), Fei Ding, [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ), [Philip H.S. Torr](http://www.robots.ox.ac.uk/~phst/)<br>
CVPR 2021.


MUSES is a large-scale video dataset, designed to spur researches on a new task called multi-shot temporal event localization. We present a baseline aproach (denoted as MUSES-Net) that achieves SOTA performance on MUSES. It also reports an mAP of 56.9% on THUMOS14 at IoU=0.5. 


The code largely borrows from [SSN][ssn] and [P-GCN][pgcn]. Thanks for their great work!

Find more resouces (e.g. annotation file, source videos) on our [project page][project-page].

# Updates
[2022.3.19] Add support for the MUSES dataset. The proposals, models, source videos of the MUSES dataset are released. Stay tuned for MUSES v2, which includes videos from more countries.<br>
[2021.6.19] Code and the annotation file of MUSES are released. Please find the annotation file on our [project page][project-page].

# Contents
----
* [Updates](#updates)
* [Usage Guide](#usage-guide)
   * [Prerequisites](#prerequisites)
   * [Data Preparation](#data-preparation)
   * [Reference Models](#reference-models)
   * [Testing Trained Models](#testing-trained-models)
   * [Training](#training)
* [Contact](#contact)

----


# Usage Guide

## Prerequisites
[[back to top](#MUSES)]

The code is based on PyTorch. The following environment is required.
- Python 3
- [PyTorch >= 1.3.0][pytorch] 
- CUDA >= 9.2

Other minor Python modules can be installed by running
```
pip install -r requirements.txt
```

The code relies on CUDA extensions. Build them with the following command:
```
python setup.py develop
```

After installing all dependecies, run `python demo.py` for a quick test.

## Data Preparation
[[back to top](#MUSES)]

We support experimenting with THUMOS14 and MUSES. The video features, the proposals and the reference models are provided on [OneDrive][onedrive].

### Features and Proposals
- THUMOS14: The features and the proposals are the same as thosed used by PGCN. 
Extract the archive `thumos_i3d_features.tar` and put the features in `data/thumos14` folder. The proposal files are already contained in the repository. We expect the following structure in this folder.
  ```text
  - data
    - thumos14
      - I3D_RGB
      - I3D_Flow
  ```


- MUSES: Extract the archives of features and proposal files.
  ```bash
  # The archive does not have a directory structure
  # We need to create one
  mkdir -p data/muses/muses_i3d_features
  tar -xf muses_i3d_features.tar -C data/muses/muses_i3d_features
  tar -xf muses_proposals.tar -C data/muses
  ```
  We expect the following structure in this folder.
  ```text
  - data
    - muses
      - muses_i3d_features
      - muses_test_proposal_list.txt
      - ...
  ```
You can also specify the path to the features/proposals in the config files `data/cfgs/*.yml`.

### Reference Models

Put the `reference_models` folder in the root directory of this code:
```
 - reference_models
   - muses.pth.tar
   - thumos14_flow.pth.tar
   - thumos14_rgb.pth.tar
```

## Testing Trained Models
[[back to top](#MUSES)]

You can test the reference models by running a single script
```
bash scripts/test_reference_models.sh DATASET
```
Here `DATASET` should be `thumos14` or `muses`.

Using these models, you should get the following performance

### MUSES

|| 0.3   | 0.4   | 0.5   | 0.6  | 0.7   | Average |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|mAP| 26.5 | 23.1 | 19.7 | 14.8 | 9.5 | 18.7  |

*Note: We re-train the network on MUSES and the performance is higher than that reported in the paper.*


### THUMOS14

|Modality|0.3|0.4|0.5|0.6|0.7|Average|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|RGB |60.14	|54.93 |46.38 |	34.96| 21.69|	43.62|
|Flow|64.64	|60.29 |53.93 |	42.84| 29.70| 50.28|
|R+F |68.93	|63.99 |56.85 |	46.25| 30.97| 53.40|



The testing process consists of two steps, detailed below.

1. Extract detection scores for all the proposals by running
```
python test_net.py DATASET CHECKPOINT_PATH RESULT_PICKLE --cfg CFG_PATH
```
Here, RESULT_PICKLE is the path where we save the detection scores. CFG_PATH is the path of config file, e.g. `data/cfgs/thumos14_flow.yml`.

2. Evaluate the detection performance by running
```
python eval.py DATASET RESULT_PICKLE --cfg CFG_PATH
```

On THUMOS14, we need to fuse the detection scores with RGB and Flow modality. This can be done by running
```
python eval.py DATASET RESULT_PICKLE_RGB RESULT_PICKLE_FLOW --cfg CFG_PATH --score_weights 1 1.2 --cfg CFG_PATH_RGB
```

## Training
[[back to top](#MUSES)]

Train your own models with the following command
```
python train_net.py  DATASET  --cfg CFG_PATH --snapshot_pref SNAPSHOT_PREF --epochs MAX_EPOCHS
```
SNAPSHOT_PREF: the path to save trained models and logs, e.g `outputs/snapshpts/thumos14_rgb/`. 

We provide a script that finishes all steps, including training, testing, and two-stream fusion. Run the script with the following command
```
bash scripts/do_all.sh DATASET
```
Note: The results may vary in different runs and differs from those of the reference models. It is encouraged to use the average mAP as the primary metric. It is more stable than mAP@0.5.

# Citation
Please cite the following paper if you feel MUSES useful to your research
```
@InProceedings{Liu_2021_CVPR,
    author    = {Liu, Xiaolong and Hu, Yao and Bai, Song and Ding, Fei and Bai, Xiang and Torr, Philip H. S.},
    title     = {Multi-Shot Temporal Event Localization: A Benchmark},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12596-12606}
}
```

# Related Projects
- [TadTR][tadtr]: Efficient temporal action detectioon (localization) with Transformer.



# Contact
[[back to top](#MUSES)]

For questions and suggestions, file an issue or contact Xiaolong Liu at "liuxl at hust dot edu dot cn".


[pytorch]:https://github.com/pytorch/pytorch
[ssn]:http://yjxiong.me/others/ssn/
[pgcn]: https://github.com/Alvin-Zeng/PGCN
[project-page]: http://songbai.site/muses/
[onedrive]: https://husteducn-my.sharepoint.com/:f:/g/personal/liuxl_hust_edu_cn/EpOGTXHbu1JHjxBgVGEa2kMBZtD5y98twz203W9hiEx2tQ?e=e0Ecxo
[tadtr]: https://github.com/xlliu7/TadTR