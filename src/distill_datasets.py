import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
import sys
sys.path.append('../')
import time

__author__ = 'Bonggun Shin'


maxlen_list = {"mr":80, "sstb":60, "sst5":60, "subj":140, "trec":40, "cr":110, "mpqa":50}


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


def get_embedding(dim, dataset_name='s17'):
    print('Loading w2v...')

    if dataset_name == 'sstb' or dataset_name == 'sst5' or dataset_name == 'cr' or dataset_name == 'mr':
        filepath = '../data/w2v/w2v-%d-amazon.gnsm' % (dim)
    else: # wsj, nytimes, wiki
        filepath = '../data/w2v/corpus.nyt+wiki+wsj.skip.d%d.bin' % (dim)

    print('[w2v] %s' % filepath)


    if filepath.endswith('.gnsm'):
        emb_model = KeyedVectors.load(filepath)
    else: # filepath.endswith('.bin')
        emb_model = KeyedVectors.load_word2vec_format(filepath, binary=True, unicode_errors='ignore')


    print('creating w2v mat...')
    word_index = emb_model.vocab
    embedding_matrix = np.zeros((len(word_index) + 1, dim), dtype=np.float32)
    for word, i in word_index.items():
        embedding_vector = emb_model[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i.index] = embedding_vector

    return embedding_matrix, emb_model.vocab


def load_s17(vocab, target='trn', pathbase='../data/sentiment_analysis/s16/'):
    x_text = [line.split('\t')[2] for line in open(pathbase + target, "r").readlines()]
    x = []
    maxlen = 0
    for s in x_text:
        one_doc = []
        tokens = s.strip().split(" ")
        maxlen = max(maxlen, len(tokens))
        for token in tokens:
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split('\t')[1]
        if senti == 'negative':
            y.append(0)
        elif senti == 'objective':
            y.append(1)
        elif senti == 'positive':
            y.append(2)

    return np.array(x), np.array(y), maxlen

def load_sst5(vocab, target='trn', pathbase='../data/sentiment_analysis/sst5/'):
    maxlen = 0
    x_text = [line.split('\t')[2] for line in open(pathbase + target, "r").readlines()]
    x = []
    for s in x_text:
        one_doc = []
        tokens = s.strip().split(" ")
        maxlen = max(maxlen, len(tokens))
        for token in tokens:
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split('\t')[1]
        if senti == 'neutral':
            y.append(2)
        elif senti == 'positive':
            y.append(3)
        elif senti == 'very_positive':
            y.append(4)
        elif senti == 'negative':
            y.append(1)
        elif senti == 'very_negative':
            y.append(0)

    return np.array(x), np.array(y), maxlen


def load_harvard_datasets(vocab, target='trn', pathbase='../data/sentiment_analysis/sstb/'):
    x_text = [line.strip().split(' ')[1:] for line in open(pathbase + target, "r").readlines()]
    x = []
    maxlen = 0
    for tokens in x_text:
        one_doc = []
        maxlen = max(maxlen, len(tokens))
        for token in tokens:
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split(' ')[0]
        y.append(int(senti))

    return np.array(x), np.array(y), maxlen


def load_sstb(vocab, target='trn', pathbase='../data/sentiment_analysis/sstb/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_trec(vocab, target='trn', pathbase='../data/sentiment_analysis/trec/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_mpqa(vocab, target='trn', pathbase='../data/sentiment_analysis/mpqa/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_cr(vocab, target='trn', pathbase='../data/sentiment_analysis/cr/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_mr(vocab, target='trn', pathbase='../data/sentiment_analysis/mr/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_subj(vocab, target='trn', pathbase='../data/sentiment_analysis/subj/'):
    return load_harvard_datasets(vocab, target=target, pathbase=pathbase)


def load_all(w2vdim, dataset_name='sst5', vocab=None):
    if vocab is None: # fresh start
        embedding, vocab = get_embedding(w2vdim, dataset_name=dataset_name)
        max_features = len(vocab)+1
    else: # validate distill
        embedding = None

    if dataset_name == 's17':
        (x_trn, y_trn, maxlen_trn) = load_s17(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_s17(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_s17(vocab, target='tst')

    elif dataset_name=='sst5':
        (x_trn, y_trn, maxlen_trn) = load_sst5(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_sst5(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_sst5(vocab, target='tst')

    elif dataset_name=='sstb':
        (x_trn, y_trn, maxlen_trn) = load_sstb(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_sstb(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_sstb(vocab, target='tst')

    elif dataset_name == 'trec':
        (x_trn, y_trn, maxlen_trn) = load_trec(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_trec(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_trec(vocab, target='tst')

    elif dataset_name == 'mpqa':
        (x_trn, y_trn, maxlen_trn) = load_mpqa(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_mpqa(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_mpqa(vocab, target='tst')

    elif dataset_name == 'cr':
        (x_trn, y_trn, maxlen_trn) = load_cr(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_cr(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_cr(vocab, target='tst')

    elif dataset_name == 'mr':
        (x_trn, y_trn, maxlen_trn) = load_mr(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_mr(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_mr(vocab, target='tst')

    else: # dataset_name == 'subj'
        (x_trn, y_trn, maxlen_trn) = load_subj(vocab, target='trn')
        (x_dev, y_dev, maxlen_dev) = load_subj(vocab, target='dev')
        (x_tst, y_tst, maxlen_tst) = load_subj(vocab, target='tst')


    maxlen = maxlen_list[dataset_name]
    print('maxlen(%d)' % maxlen)

    x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen, value=len(vocab))
    x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen, value=len(vocab))
    x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen, value=len(vocab))

    if embedding is not None: # fresh start
        return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features, maxlen

    else: # validate distill
        return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), None, None, maxlen
