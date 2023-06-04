# MonoDepth

## Setup


```shell
conda create -n ht_dcmnet python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
My experiments has been done with PyTorch 1.9.0, CUDA 11.2, Python 3.8.5 and Ubuntu 18.04. I use 1 NVIDIA RTX 3090 GPU for training.

## Simple Prediction

You can simply visualize the depth estimation results on some images from KITTI with:

```shell
python test_simple.py --image_path=./test_images/
```

You can check depth estimation results with other images from KITTI or your own datasets by adding test images on the folder named "test_images". You can run the code without GPU by using --no_cuda flag.

## KITTI Dataset
`
The dataset path on the 3090GPU is `/remote-home/share/KITTI/KITTI_fengyi/`.

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P /YOUR/DATA/PATH/
```

KITTI images are converted from `.png` to `.jpg` extension with this command for fast load times during training:

```shell
find /YOUR/DATA/PATH/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

The commands above results in the data_path:
```
/YOUR/DATA/PATH
  |----2011_09_26
      |----2011_09_26_drive_0001_sync  
          |-----.......  
          |----image_02
              |-----data
                  |-----0000000000.jpg
                  |-----.......
              |-----timestamps.txt
          |-----.......
      |----.........        
  |----2011_09_28        
  |----.........        
```

## Training

For training, you have to pre-train Swin Transformer encoder in ImageNet-1k dataset.

You can either simply download ImageNet-pretrained encoder weight [here](https://drive.google.com/drive/folders/1I3E3qLFoYeDw8pmbFbr4TNpElnAaxdtO?usp=sharing) named '104checkpoint.pth' or train Swin Transformer yourself with PyTorch offical [code](https://github.com/pytorch/examples/tree/main/imagenet).
Then, you place the pretrained weight in ./checkpoints/imagenet folder.

The depth estimation network is trained by running:
```shell
bash run.sh
```
You need to setup the parameters of `depth_encoder, use_flow, train_flow` before training.

## Evaluation

Before evaluation, you should prepare ground truth depth maps by running:

```shell
python export_gt_depth.py --data_path /YOUR/DATA/PATH --split eigen
```

The following example command evaluates best weights:

```shell
python evaluate_depth.py --data_path=/YOUR/DATA/PATH --load_weights_folder ./checkpoints/best/
```


## Reference

1. Monodepth2 - https://github.com/nianticlabs/monodepth2
2. timm - https://github.com/rwightman/pytorch-image-models
3. mmsegmentation - https://github.com/open-mmlab/mmsegmentation
