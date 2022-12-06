# OCET: One-dimensional Convolution Embedding Transformer for Stock Trend Prediction

This repo is a simple implementation of OCET on the FI-2010 dataset.

## Dataset - FI2010
```
https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master/data
```
data file is as follow:
```
/data
|   Train_Dst_NoAuction_DecPre_CF_7.txt
|   Test_Dst_NoAuction_DecPre_CF_7.txt
|   Test_Dst_NoAuction_DecPre_CF_8.txt
|   Test_Dst_NoAuction_DecPre_CF_9.txt
```

## Environment
```
pip install torch einops numpy tqdm
```

## Train
```
python train.py
```

## Others
we also privide some high-precision models in recent years (deeplob, deepfilio, deeplobattenton etc). You can get it in 'model'.