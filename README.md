# SwinShadow


## Requirement

* Python 3.10
* Pytorch 2.0.1

```
pip install -r requirements.txt
```


## Datasets

* [SBU dataset](https://pan.baidu.com/s/1T17NSC8ynJnsoWzETm2pBw) code:gxqt
* [ISTD, ISTD+ dataset](https://pan.baidu.com/s/1psdyDHDyG20VbKUlnNhkjQ) code:gxqt
* [UCF dataset](https://pan.baidu.com/s/1GzUWWvhbVOHQnDVjsYR0Zw) code:gxqt


## Training

download the pretrain Swin Transformer backbone from [here](https://pan.baidu.com/s/1l6YTVAWOLA7hZ9KduLaSeg) (code:gxqt), put it in the model folder

**1. training on sbu dataset**

modify `tool/dataset.py` line 27 and line 28, run `python sbu.py`, the ckpt will be saved in `./sbu/ckpt` folder


**2. training on istd dataset**

modify `tool/istddata.py` line 27 and line 28, run `python istd.py`, the ckpt will be saved in `./istd/ckpt` folder


## Testing

run `python tistd.py/tsbu.py/tucf.py`, the file will be saved in `results/xxx`