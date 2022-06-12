# Fine-Grained Hashing With Double Filtering

This project is a pytorch implementation of [*Fine-Grained Hashing With Double Filtering*](https://ieeexplore.ieee.org/document/9695302). 

## Requirements

python 3.7.7

numpy 1.18.4

pandas 1.0.4

Pillow 9.1.1

torch 1.5.0+cu101

torchvision 0.6.0+cu101

## Datasets Prepare

Download corresponding dataset, and move the folder that contains all images to the corresponding folder in "datasets".

## Training 

```shell
sh train.sh
```

## Citation

```
@article{ChenLWGX_TIP22,
  author    = {Zhen{-}Duo Chen and Xin Luo and Yongxin Wang and Shanqing Guo and Xin{-}Shun Xu},
  title     = {Fine-Grained Hashing With Double Filtering},
  journal   = {{IEEE} Trans. Image Process.},
  volume    = {31},
  pages     = {1671--1683},
  year      = {2022},
  doi       = {10.1109/TIP.2022.3145159},
}
```
