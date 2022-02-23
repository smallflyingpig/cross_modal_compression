# cross_modal_compression
officical repository for ACM MM 2021 paper: "Cross Modal Compression: "Cross Modal Compression: Towards Human-comprehensible Semantic Compression"


# Prerequisites
 - linux
 - python 3.5 (not test on other versions)
 - pytorch 1.3+
 - torchaudio 0.3
 - librosa, pysoundfile
 - json, tqdm, logging



## data preparing
### download dataset and pretrained model
 - you can download CUB-200-2011 dataset and MS COCO 2014 from the offficial site
 - download our json file for MS COCO from here([google drive](https://drive.google.com/drive/folders/1G5EU9w5t7SaYMiuY4FHW9gJKF1u2oa0T?usp=sharing), [百度网盘](https://pan.baidu.com/s/13QFoVABF4Z3cNYj2u4VBeQ)提取码：c31g)
 - download our pretrained models from here([google drive](https://drive.google.com/drive/folders/11x85FXGBMyQoB5Tl2bn6GqBotpAxDM7B?usp=sharing), [百度网盘](https://pan.baidu.com/s/17npC-FPgzocmbs_Eihgx2A)提取码：i0un)

## training the CMC
```
export PYTHONPATH=path_for_this_project
```
### training the DAMSM model
```
# for CUB dataset
python ./TextImage/pretrain_DAMSM.py --cfg ./cfg/bird_DAMSM.yml --data_dir ./data/birds --dataset bird --output_dir ./output/TextImage --no_dist
# for COCO
python ./TextImage/pretrain_DAMSM.py --cfg ./cfg/coco_DAMSM.yml --data_dir ./data/coco --dataset coco --output_dir ./output/TextImage --no_dist
```
### training the ImageText model
```
# for CUB
python ./ImageText/train.py --cfg ./cfg/bird_train.yml --data_dir ./data/birds --dataset bird --output_dir ./output/ImageText
# for COCO
python ./ImageText/train.py --cfg ./cfg/coco_train.yml --data_dir ./data/coco --dataset coco --output_dir ./output/ImageText
```
### training the TextImage model
```
# for CUB
# first set the text encoder path in ./cfg/bird_train.yml: TRAIN.NET_E
python ./TextImage/train.py --cfg ./bird_train.yml --data_dir ./data/birds --dataset bird --output_dir ./output/TextImage
# for COCO
# first set the text encoder path in ./cfg/coco_train.yml: TRAIN.NET_E
python ./TextImage/train.py --cfg ./coco_train.yml --data_dir ./data/coco --dataset coco --output_dir ./output/TextImage
```
## eval the CMC
1. write the pretrained models' paths in cfg/bird_eval.yml for CUB-200-2011 dataset or cfg/coco_eval.yml for MS COCO dataset
2. run
```
python ./ImageText/end_to_end_test.py --cfg cfg/coco_eval.yml --data_dir COCO_PATH --output_dir ./output/end_to_end_coco_test
```
for MS COCO or 
```
python ./ImageText/end_to_end_test.py --cfg cfg/coco_eval.yml --data_dir CUB_PATH --output_dir ./output/end_to_end_bird_test
```


# project home page
https://smallflyingpig.github.io/cross_modal_compression_mainpage/main.html
Feel free to mail me at: jiguo.li@vipl.ict.ac.cn/jgli@pku.edu.cn, if you have any question about this project.

# Acknowledgement
Thanks to the valuable discussion with Junlong Gao. Besides, thanks to the open source of [COCO API](https://github.com/cocodataset/cocoapi), [AttnGAN](https://github.com/taoxugit/AttnGAN), [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

 **Note that this work is only for research. Please do not use it for illegal purposes.**


