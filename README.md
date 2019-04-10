# Learning to Forecast and Refine Residual Motion for Image-to-Video Generation (ECCV 2018)

This repository holds the Pytorch implementation of [Learning to Forecast and Refine Residual Motion for Image-to-Video Generation](https://arxiv.org/abs/1807.09951) by Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia and Dimitris Metaxas. If you find our code useful in your research, please consider citing:

```
@inproceedings{zhaoECCV18learning,
  author    = {Zhao, Long and Peng, Xi and Tian, Yu and Kapadia, Mubbasir and Metaxas, Dimitris},
  title     = {Learning to forecast and refine residual motion for image-to-video generation},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {387--403},
  year      = {2018}
}
```

## Introduction

We propose a two-stage generative framework where videos are forecasted from structures and then refined by temporal signals. In the forecasting stage, to model motions more efficiently, we train networks to learn residual motion between the current and future frames, which avoids learning motion-irrelevant details. In the refining stage, to ensure temporal consistency, we build networks upon spatiotemporal 3D convolutions. The code for training and testing our approach for facial expression retargeting on the [MUG datatset](https://mug.ee.auth.gr/fed/) is provided in this repository.

**Note that the current version only contains the code for the forecasting stage. We are working on an improved version of the refining stage, which will come very soon.**

We utilize [3DFFA](https://github.com/cleardusk/3DDFA) to compute 3DMM for all face images in the dataset. Please refer to the corresponding part in [our paper]((https://arxiv.org/abs/1807.09951)) and [the repository of 3DFFA](https://github.com/cleardusk/3DDFA) for more details.

## Quick start

This repository is build upon Python v2.7 and Pytorch v1.0.1. The code may also work with Pytorch v0.4.1 but has not been tested.

### Installation

1. Clone this repository. In [Google Drive](https://drive.google.com/drive/folders/1U-Xp7w31jJJ3IN98ZfEHyXVqCQNXlG3s?usp=sharing), download `param_mesh.mat` and put it into `configs` directory. Then download `phase1_wpdc_vdc_v2.pth.tar` and `shape_predictor_68_face_landmarks.dat`, and put them into `models` directory.

	```
	>> git clone git@github.com:garyzhao/FRGAN.git
	>> cd FRGAN
	```

2. We recommend installing Python v2.7 from [Anaconda](https://www.anaconda.com/), installing Pytorch (>= 1.0.1) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. In addition, you need to install dependencies below.

	```
	>> pip install -r requirements.txt
	```

3. Build the C++ extension for computing normal maps from the 3D face model.

	```
	>> cd dfa
	>> python setup.py build_ext --inplace
	>> cd -
	```
	
### Data preparation

Download [MUG datatset](https://mug.ee.auth.gr/fed/) and organize data like this

```
MUG
|-- 001
    |-- anger
        |-- take000
            |-- img_0000.jpg
            |-- img_0001.jpg
            |-- img_0002.jpg
            |-- ...
        |-- take001
            |-- img_0000.jpg
            |-- img_0001.jpg
            |-- img_0002.jpg
            |-- ...
        |-- ...
    |-- disgust
        |-- take000
            |-- img_0000.jpg
            |-- img_0001.jpg
            |-- img_0002.jpg
            |-- ...
        |-- ...
    |-- ...
|-- 002
    |-- ...
|-- ...
```

Then run the following script in the project directory to preprocess the data

```
>> python datasets/mug_process_dataset.py --inp_path $MUG_ROOT_PATH$ --out_path datasets/mug --out_size 64
```

Replace `$MUG_ROOT_PATH$` by the path to the downloaded MUG directory. The preprocessed data will be saved in `datasets/mug/mug64` directory. You can obtain results with higher resolutions by changing the `out_size` parameter (e.g., 64, 96 or 128).

### Training

To train the network, try the following script in the project directory:

```
>> export PYTHONPATH=".:$PYTHONPATH"
>> export CUDA_VISIBLE_DEVICES=0,1
>> python mug_train_forecast.py --img_dir_path datasets/mug/mug64/ --batch_size 64 --num_epochs 100 --snapshot 2
```

Please modify `CUDA_VISIBLE_DEVICES` and `batch_size` according to your GPU settings. You may also change the value of `img_size` (64 by default) and `h_dim` (128 by default) if you train on images with higher resolutions. Please refer to `mug_train_forecast.py` for more details.

### Testing

To test the network, try:

```
>> python mug_test.py --img_dir_path datasets/mug/mug64/ --out_dir_path examples --checkpoint $CHECKPOINT_PATH$
```

Replace `$CHECKPOINT_PATH$` by the path to the checkpoint saving during training.