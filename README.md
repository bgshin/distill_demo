
## Requirements


```
pip install tensorflow
pip install keras
pip install gensim
pip install sklearn
```

## Train a teacher

```
cd src/
python train_teacher.py -ds sst5 -m cnn2
```

## Train an autoencoder

```
cd src/
python train_ae.py
```