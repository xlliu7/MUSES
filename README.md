# MUSES
This repo holds the code and the models for MUSES, introduced in the paper:<br>
Multi-shot Temporal Event Localization: a Benchmark<br>
[Xiaolong Liu](https://github.com/xlliu7), [Yao Hu](https://scholar.google.com/citations?user=LIu7k7wAAAAJ), [Song Bai](http://songbai.site), Fei Ding, [Xiang Bai](http://122.205.5.5:8071/~xbai/), [Philip H.S. Torr](http://www.robots.ox.ac.uk/~phst/)<br>
CVPR 2021.


MUSES is a large-scale video dataset, designed to spur researches on a new task called multi-shot temporal event localization. We present a baseline aproach that achieves SOTA peformance on MUSES. It also reports an mAP of 56.9% on THUMOS14 at IoU=0.5. 

Refer to the [paper](https://arxiv.org/abs/2012.09434) and the [project page](http://songbai.site/muses/) for more information.

# Updates
[2021.6.19] Code and the annotation file of MUSES is released. Please find the annotation file on our [project page](http://songbai.site/muses/).

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

We support experimenting with THUMOS14. The support for MUSES will come soon. To run the experiments, you can directly download the pre-extracted features.

- THUMOS14: The features are provided by PGCN. You can download them from [[OneDrive]](https://husteducn-my.sharepoint.com/:u:/g/personal/liuxl_hust_edu_cn/EQ-5j4yQL0pNmgV4N0UPiokBFE3BX2TWEAzUxqNaAp2xEw?e=2SkUdn) (2.4G).
Extract the archive and put the features in `data/thumos14` folder. We expect the following structure in this folder.
```text
- data
  - thumos14
    - I3D_RGB
    - I3D_Flow
```


## Reference Models
[[back to top](#MUSES)]

Download models trained by us and put them in the `reference_models` folder:
- THUMOS14: Models trained with RGB and Flow. [[OneDrive]](https://husteducn-my.sharepoint.com/:f:/g/personal/liuxl_hust_edu_cn/Ev6jpwGyKklHgCKwRwNEpaEB7FsRE_CmS-0sXkdaNgPPcw?e=b0BnpC)

## Testing Trained Models
[[back to top](#MUSES)]

You can test the reference models and fuse different modalities on THUMOS14 by running a single script
```
bash scripts/test_reference_models.sh
```

Using these models, you should get the following performance

||RGB|Flow|RGB+Flow|
|:-:|:-:|:-:|:-:|
|mAP@IoU=0.5|46.4|53.9|56.9|

The results with RGB+Flow at all IoU thresholds
```
0.10   | 0.20   | 0.30   | 0.40   | 0.50   | 0.60   | 0.70   | 0.80   | 0.90   | Average |
0.7377 | 0.7219 | 0.6893 | 0.6399 | 0.5685 | 0.4625 | 0.3097 | 0.1334 | 0.0192 | 0.4758  
```

The testing process consists of two steps, detailed below.

1. Extract detection scores for all the proposals by running
```
python test_net.py DATASET CHECKPOINT_PATH RESULT_PICKLE --cfg CFG_PATH
```
Here, DATASET should be `thumos14` or `muses`. RESULT_PICKLE is the path where we save the detection scores. CFG_PATH is the path of config file, e.g. `data/cfgs/thumos14_flow.yml`.

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
python train_net.py  DATASET  --cfg CFG_PATH --snapshot_pref SNAPSHOT_PREF --epochs 20
```
SNAPSHOT_PREF: the path to save trained models and logs, e.g `outputs/snapshpts/thumos14_rgb/`. 

We provide a script that finishes all steps on THUMOS14, including training, testing, and two-stream fusion. Run
```
bash scripts/do_all.sh
```

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
- [TadTR](https://github.com/xlliu7/TadTR): Temporal action detectioon (localization) with Transformer.

# Contact
[[back to top](#MUSES)]

For questions and suggestions, file an issue or contact Xiaolong Liu at "liuxl at hust dot edu dot cn".


[thumos14]:http://crcv.ucf.edu/THUMOS14/download.html
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[anet_down]:https://github.com/activitynet/ActivityNet/tree/master/Crawler
[map]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
[action_kinetics]:http://yjxiong.me/others/kinetics_action/
[pytorch]:https://github.com/pytorch/pytorch
[ssn]:http://yjxiong.me/others/ssn/
[emv]:https://github.com/zbwglory/MV-release
[features_google]: https://drive.google.com/open?id=1C6829qlU_vfuiPdJSqHz3qSqqc0SDCr_
[features_baidu]: https://pan.baidu.com/s/1Dqbcm5PKbK-8n0ZT9KzxGA
