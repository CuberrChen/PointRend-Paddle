# PointRend-Paddle

English | [简体中文](./README_cn.md)
   
  * [PointRend](#pointrend)
      * [1 Introduction](#1-introduction)
      * [2 Metrics](#2-Metrics)
      * [3 Dataset](#3-dataset)
      * [4 Environment](#4-environment)
      * [5 Quick Start](#5-quick-start)
         * [Step1: Install](#step0-Install)
         * [Step1: Clone](#step1-clone)
         * [Step2: Training](#step2-training)
         * [Step3: val](#step3-val)
         * [Use Pre-trained Models to Infer](#use-pre-trained-models-to-infer)
      * [6 Code Structure and Explanation](#6-code-structure-and-explanation)
         * [6.1 Code Structure](#51-code-structure)
         * [6.2 Parameter Explanation](#52-parameter-explanation)
         * [6.3 Training Process](#53-training-process)
            * [One GPU Training](#one-gpu-training)
            * [Multiple GPUs Training](#multiple-gpus-training)
      * [7 Model Information](#7-model-information)
      * [8 Customization](#8-customization)

## 1 Introduction

Paddle version of Paper“PointRend: Image Segmentation as Rendering(CVPR2020)”.

This project uses Baidu's paddlepaddle framework to reproduce the CVPR2020 paper's model PointRend. **Note: only the semantic segmentation experiment of Semantic FPN + PointRend on the cityscapes dataset is done here, excluding the instance segmentation experiment of Maskrcnn + Pointrend. The correctness of PointRend based on paste reproduction is verified.**

The project relies on the paddleseg tool.

**PointRend With Seg Architecture:**

![PointRend](./images/pointrend.png)

![PointRend Result](./images/pointrendresult.png)


**Paper:** [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

## 2 Metrics

| Model                   | mIOU |
| ----------------------- | -------- |
| SemanticFPN+PointRend(paper-Pytorch)     | 78.5     |
| SemanticFPN+PointRend(ours-Paddle) | 78.78  |

## 3 Dataset

The dataset is [Cityscapes](https://www.cityscapes-dataset.com/)

- The size of dataset: There are 19 categories, and 5000 images are of 1024*2048 pixels in width and height
  - Training set: 2975 images
  - Validation set: 500 images
  - Test set: 1525 images

data should be located at data/

```
data/
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── leftImg8bit
│   │   ├── test
│   │   │   ├── berlin
│   │   │   ├── ...
│   │   │   └── munich
│   │   ├── train
│   │   │   ├── aachen
│   │   │   ├── ...
│   │   │   └── zurich
│   │   └── val
│   │       ├── frankfurt
│   │       ├── lindau
│   │       └── munster
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt

```
.txt format style like as follow:

```leftImg8bit/test/mainz/mainz_000001_036412_leftImg8bit.png,gtFine/test/mainz/mainz_000001_036412_gtFine_labelTrainIds.png```

which can achieved by use PaddleSeg's create_dataset_list.py(need to clone PaddleSeg from PaddleSeg's git repo firstly):
 
```
python PaddleSeg/tools/create_dataset_list.py ./data/cityscapes/ --type cityscapes --separator ","

```

## 4 Environment

- Hardwares: XPU, GPU, CPU
- Framework: 
  - PaddlePaddle >= 2.0.2

## 5 Quick Start

The project is developed based on Paddleseg. Except that `train.py` is modified, other `val.py` and `predict.py` are the same as Paddleseg. The model and user-defined loss function definitions are located in the `paddleseg/models` directory.

### install(cmd line)

```bash
pip install -r requirements.txt
``` 

### step1: clone 

``` bash
# clone this repo(Note: maybe need to checout branch after git clone)
git clone git@github.com:CuberrChen/PointRend-Paddle.git
```

### Step2: Training

The training adopts the warmup learning rate strategy opened by default and the momentum optimizer. See line 181 in `train.py`. If closed, use the policy in `.yml` .
``` bash
# V100*4
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m paddle.distributed.launch train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml --num_workers=16 --use_vdl --do_eval --save_interval 1000 --save_dir output
```
```
# single V100 (I haven't tried it yet, so you need to adjust the learning rate, iters and batchsize according to the specific configuration)

python train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml--num_workers 4 --use_vdl --do_eval --save_interval 1000 --save_dir output --batch_size 4

```

### Step3: Eval

The default path of the pre training model is'output/best_model/model.pdparams'

```bash
# eval  
CUDA_VISIBLE_DEVICES=0 
python val.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml --model_path output/best_model/model.pdparams
```


### Use Pre-trained Models to Infer
The Pre-trained model is used to test the image, For specific use, please refer to [Paddleseg doc](https://paddleseg.readthedocs.io/zh_CN/release-2.1/index.html)

The use example is as follows:
```bash
# Use Pre-trained Models to Infer
python predict.py \
       --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml \
       --model_path output/best_model/model.pdparams \
       --image_path data/xxx/JPEGImages/0003.jpg \
       --save_dir output/result
```

## 6 Code Structure and Explanation

### 6.1 Code Structure

```
├── README.md
├── README_EN.md
├── images/ # save images for README
├── data/ #data path
├── paddleseg/ # paddleseg tool include models/loss definition
├── utils/ # tools
├── lr_scheduler/ # scheduler defined by self
├── output/ # output path
├── run.sh # AIStudio 4 card training shell  
├── ...
├── train.py 
├── eval.py 
└── predict.py 
```

### 6.2 Parameter Explanation
For specific parameter settings (mainly the modification of config file), please refers to [Paddleseg doc](https://paddleseg.readthedocs.io/zh_CN/release-2.1/index.html)

The only thing to note here is that the parameters of warmup are temporarily viewed in `train.py`.

the parameter setting of the model(You can enter parameter values in the config file) should refer to `paddleseg/models/pointrendseg.py`. Users need to refer to the paper to know the meaning of this part.

### 6.3 Training Process

#### One GPU Training
```
# single V100 (I haven't tried it yet, so you need to adjust the learning rate, iters and batchsize according to the specific configuration)

python train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml--num_workers 4 --use_vdl --do_eval --save_interval 1000 --save_dir output --batch_size 4

```

#### Multiple GPUs Training
``` bash
# V100*4
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m paddle.distributed.launch train.py --config configs/pointrendfpn/pointrend_resnet101_os8_cityscapes_512×1024_80k.yml --num_workers=16 --use_vdl --do_eval --save_interval 1000 --save_dir output
```


## 7 Model Information

Please refer to the following list to check other models’ information

| Information Name | Description |
| --- | --- |
| Announcer | xbchen |
| Time | 2021.08 |
| Framework Version | Paddle 2.0.2 |
| Application Scenario | Image Segmentation |
| Supported Hardwares | XPU GPU CPU |
| Download Links | [PointRendFPN: 提取码：b8ai](https://pan.baidu.com/s/1RXgac1j1bYn76Yx0fTbQfw)|
| Online Running |[AIStudio notebook](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2298566)|

## 8 Customization

Special thanks to the platform and resources provided by Baidu paddle.

**SemanticFPN+PointRend Model analysis**：

- 80000 iter,batch_size=16 for 4 GPUs(4 imgs for per gpu),base_lr=0.01 warmup+poly,**SemanticFPN+PointRend with ResNet101 's best mIOU=78.78 at Cityscaps VAL dataset**.
***Note: the reason for adopting this scheme is that the 4 cards 32g environment provided by aistudio allows 1024 × 512 enter the maximum batch_size can't reach 32(paper's setting). If the memory is enough / multiple cards are used, the parameters provided by the author are recommended. The trained model has a link at the bottom. The training code and train_0.log (79.6miou complete training log can be find in output/) have been uploaded to the repo* 

Refrence:
- [Paper Official PyTorch](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend)
