# Introduction
* Official Implementation of the paper, "The Pupil Has Become the Master: Teacher-Student Model-Based Word Embedding Distillation with Ensemble Learning", B. Shin, H. Yang, and J.D. Choi, IJCAI 2019.
* Author: Bonggun Shin

## Download Data

* Unzip [Data](https://drive.google.com/file/d/1YcefkdeaXHDlTHJcdmuT6u49EZjEcjR4/view?usp=sharing) into 'data' folder

```
distill_demo/data/sentiment_analysis/*
distill_demo/data/w2v/*
```

## Requirements

```
pip install tensorflow
pip install keras
pip install gensim
pip install sklearn
```

## Train teachers

```
cd src/
python train_teacher.py -ds sst5 -m cnn2 -t 0
python train_teacher.py -ds sst5 -m cnn2 -t 1
...
python train_teacher.py -ds sst5 -m cnn2 -t 9

python train_teacher.py -ds sst5 -m lstm -t 0
python train_teacher.py -ds sst5 -m lstm -t 1
...
python train_teacher.py -ds sst5 -m lstm -t 9
```

## Train an autoencoder

```
cd src/
python train_ae.py
```

## Distill without ensemble

```
cd src/
python distill.py
```

## Distill with ensemble

```
cd src/
python extract_logits.py
python distill.py
```