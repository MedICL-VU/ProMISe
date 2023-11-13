# ProMISe
ProMISe: **Pro**mpt-driven  3D **M**edical **I**mage **Se**gmentation Using Pretrained Image Foundation Models
```
@article{li2023promise,
  title={Promise: Prompt-driven 3D Medical Image Segmentation Using Pretrained Image Foundation Models},
  author={Li, Hao and Liu, Han and Hu, Dewei and Wang, Jiacheng and Oguz, Ipek},
  journal={arXiv preprint arXiv:2310.19721},
  year={2023}
}
```

---------------------------------
**Recent news**

(11/12/23) The code is uploaded and updated.

---------------------------------
**Installation**
```
conda create -n promise python=3.9
conda activate promise
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 # install pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git # install segment anything packages
pip install git+https://github.com/deepmind/surface-distance.git # for normalized surface dice (NSD) evaluation
pip install -r requirements.txt
```

**Training**
```
python train.py --data colon --data_dir your_data_directory --save_dir to_save_model_and_log
```

**Test**

```
python test.py --data colon --data_dir your_data_directory --save_dir to_save_model_and_log --split test
```


use [pretrained ProMISe](https://drive.google.com/drive/folders/1Yol2tIaNYVve6JQ3osg2pjDRgwVeS-IF?usp=sharing).
--use_pretrain --pretrain_path /your_downladed_path/colon_pretrain_promise.pth


**Tips**

- Set "num_worker" based on your cpu to boost the data loading speed, it matters. From my device, loading data takes 30 seconds if num_workers = 1.
- please specify the save_name.
- don't forget to download the pretrained SAM model from [SAM-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and set the path as "checkpoint_sam".
- set "save_prediction" and "save_base_dir" if you want to save inference predictions.

- more details can be viewed in /config/config_args.py



TODO:
1. build this page for better instructions.
2. Pytorch DistributedDataParallel.

---------------------------------


Please shot an email to hao.li.1@vanderbilt.edu for any questions and always happy to help! :)


we use colon and pancreas datasets from [MSD](http://medicaldecathlon.com/) in our experiments. Preprocessed datasets are obtained from [3DSAM-adapter](https://github.com/med-air/3DSAM-adapter/).

We thank authors to share their code and data preprocess steps. 

