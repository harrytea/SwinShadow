# SwinShadow

> Official PyTorch implementation for TOMM24 "SwinShadow: Shifted Window for Ambiguous Adjacent Shadow Detection"

## Requirement

* Python 3.10
* Pytorch 2.0.1

```
pip install -r requirements.txt
```


## Datasets

* [SBU dataset](https://pan.baidu.com/s/13FFl5132EolFbeorO0TyHQ?pwd=gxqt) code:gxqt
* [ISTD, ISTD+ dataset](https://pan.baidu.com/s/1N6ZchTsrR8CizFBKyfrIQw?pwd=gxqt) code:gxqt
* [UCF dataset](https://pan.baidu.com/s/1v-Y_q3QgS6h3g_pjezeCVg?pwd=gxqt) code:gxqt


## Training

download the pretrain Swin Transformer backbone from [here](https://pan.baidu.com/s/1Vav0KJ0LAD5A5zfblGerQw?pwd=gxqt) (code:gxqt), put it in the model folder

**1. training on sbu dataset**

modify `tool/dataset.py` line 27 and line 28, run `python sbu.py`, the ckpt will be saved in `./sbu/ckpt` folder


**2. training on istd dataset**

modify `tool/istddata.py` line 27 and line 28, run `python istd.py`, the ckpt will be saved in `./istd/ckpt` folder


## Testing

run `python tistd.py/tsbu.py/tucf.py`, the file will be saved in `results/xxx`
