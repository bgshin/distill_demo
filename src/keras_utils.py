from keras.callbacks import ModelCheckpoint

__author__ = 'Bonggun Shin'


class AccCallback(ModelCheckpoint):
    def __init__(self, filepath, data, monitor='val_acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, real_save=True):

        super(AccCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.y_trn, self.x_dev, self.y_dev, self.x_tst, self.y_tst = data
        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.filepath = filepath
        self.real_save = real_save

    def evaluate(self, model):
        score_trn = model.evaluate(self.x_trn, self.y_trn, batch_size=200)[1]
        score_dev = model.evaluate(self.x_dev, self.y_dev, batch_size=200)[1]
        score_tst = model.evaluate(self.x_tst, self.y_tst, batch_size=200)[1]

        if self.score_dev < score_dev:
            self.score_trn = score_trn
            self.score_dev = score_dev
            self.score_tst = score_tst
            print("\nupdated!!")
            if self.real_save is True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        print('\n[This Epoch]')
        print('\t'.join(map(str, [score_trn, score_dev, score_tst])))
        print('[Current Best]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate(self.model)

