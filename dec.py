from keras.initializers import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.cluster import KMeans

import os
import numpy as np
import sys
import utils


class ClusterLayer(Layer):

    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha

        # k-means cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusterLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self, args, n_clusters, alpha=1.0, cluster_centroid=None):
        '''
        :param n_clusters: number of cluster(classes)
        :param alpha: soft-clustering hyper parameter.
        :param cluster_centroid: centroid each cluster.
        '''

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = args.embedding_size
        self.alpha = alpha
        self.pretrained_weights_path = os.path.join(args.save_weight_path, "ae_weights_{}.h5".format(args.dataset))
        self.cluster_centroid = cluster_centroid
        self.batch_size = args.dec_batch_size
        self.learning_rate = args.dec_lr

        # encoder layer dimensions. Decoder dimension is the opposite.
        self.encoders_dims = [self.input_dim, 500, 500, 2000, 50]
        self.input_layer = Input(shape=(self.input_dim,), name='input')
        self.dropout_fraction = 0.2

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i in range(1, len(self.encoders_dims)):
            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'selu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,
                            input_shape=(self.encoders_dims[i - 1],),
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                            bias_initializer='zeros', name='encoder_dense_%d' % i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'selu'
            decoder = Dense(self.encoders_dims[i - 1], activation=decoder_activation,
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                            bias_initializer='zeros',
                            name='decoder_dense_%d' % decoder_index)
            self.decoders.append(decoder)

            autoencoder = Sequential([
                Dropout(self.dropout_fraction, input_shape=(self.encoders_dims[i - 1],),
                        name='encoder_dropout_%d' % i),
                encoder,
                Dropout(self.dropout_fraction, name='decoder_dropout_%d' % decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for fine-tuning
        # Note that at this point dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if cluster_centroid is not None:
            assert cluster_centroid.shape[0] == self.n_clusters
            assert cluster_centroid.shape[1] == self.encoder.layers[-1].output_dim

        if os.path.isfile(self.pretrained_weights_path):
            self.autoencoder.load_weights(self.pretrained_weights_path)
            print("Load pre-trained AE")

    def p_mat(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, args, x_data, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if not os.path.isfile(self.pretrained_weights_path):

            iters_per_epoch = int(len(x_data) / self.batch_size)
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)
            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            print('layer-wise pre-train')
            current_input = x_data
            [train_log, lr_schedule, checkpoint, early_stopping] = utils.get_callbacks_ae(args)

            # greedy-layer wise training
            for i, autoencoder in enumerate(self.layer_wise_autoencoders):
                if i > 0:
                    weights = self.encoders[i - 1].get_weights()
                    dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                        activation='selu', weights=weights,
                                        name='encoder_dense_copy_%d' % i)
                    encoder_model = Sequential([dense_layer])
                    encoder_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
                    current_input = encoder_model.predict(current_input)

                autoencoder.fit(current_input, current_input,
                                batch_size=self.batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule])
                self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(
                    autoencoder.layers[-1].get_weights())

            print('Fine-tuning auto-encoder')

            # update encoder and decoder weights:
            self.autoencoder.fit(x_data, x_data,
                                 batch_size=self.batch_size,
                                 epochs=finetune_epochs,
                                 callbacks=[train_log, lr_schedule, checkpoint, early_stopping])

        else:
            print('Loading pre-trained weights for auto-encoder.')
            self.autoencoder.load_weights(self.pretrained_weights_path)

        # update encoder, decoder

        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        if self.cluster_centroid is None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(self.encoder.predict(x_data))
            self.cluster_centroid = kmeans.cluster_centers_

        # initial centroid using K-mean
        self.dec_model = Sequential([self.encoder,
                                     ClusterLayer(self.n_clusters, weights=self.cluster_centroid, name='dec')])
        self.dec_model.compile(loss='kullback_leibler_divergence', optimizer=Adam(lr=0.0001))
        return

    def cluster(self, args, x_data, y_data=None, test="train", tol=0.01, iter_max=1e6, **kwargs):

        save_path = os.path.join(args.save_weight_path, "dec_weights_{}.h5".format(args.dataset))

        if os.path.isfile(save_path):
            self.dec_model.load_weights(save_path)
            print('Restored Model weight')

        if test=="test":
            y_pred = self.dec_model.predict(x_data, verbose=0).argmax(1)
            acc = utils.cluster_acc(y_data, y_pred)
            print('Accuracy ' + str(np.round(acc, 5)))
            return

        update_interval = x_data.shape[0] / self.batch_size
        print('Update interval', update_interval)

        train = True
        iteration, index = 0, 0
        current_acc = 0
        self.accuracy = 0

        while train:
            sys.stdout.write('\r')
            # cut off iteration
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.dec_model.predict(x_data, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = (np.sum((y_pred == self.y_pred)).astype(np.float32) / y_pred.shape[0])
                if y_data is not None:
                    current_acc = utils.cluster_acc(y_data, y_pred)
                    print('Iteration ' + str(iteration) + ', Accuracy ' + str(np.round(current_acc, 5)))

                else:
                    print(str(np.round(delta_label * 100, 5)) + '% change in label assignment')

                if delta_label < tol:
                    print('Reached tolerance threshold.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                # weight changes if current
                if self.accuracy < current_acc:
                    for i in range(len(self.encoder.layers)):
                        self.encoder.layers[i].set_weights(self.dec_model.layers[0].layers[i].get_weights())
                    self.cluster_centroid = self.dec_model.layers[-1].get_weights()[0]

                    # save checkpoint
                    self.dec_model.save(save_path)
                    self.accuracy = current_acc
                    print("update weight and save checkpoint")

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index + 1) * self.batch_size > x_data.shape[0]:
                loss = self.dec_model.train_on_batch(x_data[index * self.batch_size::],
                                                     self.p[index * self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                loss = self.dec_model.train_on_batch(x_data[index * self.batch_size:(index + 1) * self.batch_size],
                                                     self.p[index * self.batch_size:(index + 1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1

            iteration += 1
            sys.stdout.flush()

        return