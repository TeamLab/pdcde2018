from keras.models import Model
from keras.optimizers import *
from keras.layers import *
from keras.initializers import *
from keras import objectives

import keras.backend as K
import tensorflow as tf


class NceLogit(Layer):

    """
    implementation Tensorflow nce-loss function.
    original source code of Tensorflow :
    https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/ops/nn_impl.py
    original paper : http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
    """

    def __init__(self, target, vocab_size, num_true, embedding_size, num_sampled, **kwargs):
        super(NceLogit, self).__init__(**kwargs)
        self.target = target
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.num_true = num_true

    def build(self, input_shape):

        self.input_batch = input_shape[0]
        self.input_dim = input_shape[1]

        self.W = self.add_weight(shape=(self.vocab_size, self.embedding_size),
                                 initializer=TruncatedNormal(mean=0.0, stddev=1.0/self.embedding_size),
                                 name='softmax_weights')

        self.bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', name='softmax_bias')

        self.built = True

    def call(self, inputs, training=None):

        target = K.cast(self.target, "int64")

        # label_flat shape [batch_size * num_true] tensor
        label_flat = K.reshape(target, [-1])
        sampled_value = tf.nn.log_uniform_candidate_sampler(true_classes=target,
                                                            num_true=self.num_true, num_sampled=self.num_sampled,
                                                            unique=True, range_max=self.vocab_size)

        # sampled shape : [num_sampled] tensor
        sampled, true_expected_count, sampled_expected_count = (K.stop_gradient(s) for s in sampled_value)
        sampled = K.cast(sampled, tf.int64)

        all_ids = K.concatenate([label_flat, sampled], axis=0)

        # Retrieve the true weights and the logits of the sampled weights.
        # weights shape is [vocab_size, embedding_size]
        all_w = tf.nn.embedding_lookup(self.W, all_ids, partition_strategy='mod')

        # true_w shape is [batch_size * num_true, embedding_size]
        true_w = K.slice(all_w, [0, 0], K.stack([K.shape(label_flat)[0], -1]))
        sampled_w = K.slice(all_w, K.stack([K.shape(label_flat)[0], 0]), [-1, -1])

        # inputs has shape [batch_size, embedding_size]
        # sampled_w has shape [num_sampled, embedding_size]
        # Apply matmul, which yields [batch_size, num_sampled]
        sampled_logits = K.dot(inputs, K.transpose(sampled_w))

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_b = tf.nn.embedding_lookup(self.bias, all_ids, partition_strategy='mod')

        # true_b is a [batch_size * num_true] tensor
        # sampled_b is a [num_sampled] float tensor
        true_b = K.slice(all_b, [0], K.shape(label_flat))
        sampled_b = K.slice(all_b, K.shape(label_flat), [-1])

        # inputs shape is [batch_size, embedding_size]
        # true_w shape is [batch_size * num_true, embedding_size]
        # row_wise_dots is [batch_size, num_true, embedding_size]
        dim = K.shape(true_w)[1:2]
        new_true_w_shape = K.concatenate([[-1, self.num_true], dim], axis=0)
        row_wise_dots = Multiply()([K.expand_dims(inputs, axis=1), K.reshape(true_w, new_true_w_shape)])

        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = K.reshape(row_wise_dots, K.concatenate([[-1], dim], 0))

        # true_logits = [batch_size, num_true]
        true_logits = K.reshape(K.sum(dots_as_matrix, axis=1), [-1, self.num_true])
        true_b = K.reshape(true_b, [-1, self.num_true])
        true_logits += true_b

        sampled_logits += sampled_b

        # out_logits = [batch_size, num_true+num_sampled]
        out_logits = K.concatenate([true_logits, sampled_logits], axis=1)

        return out_logits

    def compute_output_shape(self, input_shape):
        return tuple([self.input_batch, self.num_true+self.num_sampled])

    def get_config(self):
        config = {
            'target': self.target,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'num_sampled': self.num_sampled,
            'num_true': self.num_true
        }
        base_config = super(NceLogit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_model(args, vocab_size, embedding_size, num_document, summary=True, num_true=1):

    def nce_loss(y_true_, y_pred):
        """
        :param y_true: y_true does not use because we calculate true logits and smaple logits using NceLogit.
        :param nce_logit: nce_logit consist of [batch_size, num_true+num_sampled].
                          [batch_size, 0] is true_logits and [batch_size, 1:] is sample_logits
        :return: binary-crossentropy
        """
        true_logit = K.expand_dims(y_pred[:, num_true-1], axis=-1)
        sample_logit = y_pred[:, num_true:]

        y_true = K.concatenate([K.ones_like(true_logit), K.zeros_like(sample_logit)], axis=1)
        loss = K.mean(objectives.binary_crossentropy(y_true, K.sigmoid(y_pred)))

        return loss

    # context inputs size [batch_size* window_size * 2]
    context_inputs = Input(shape=(args.window_size*2,), name='context_inputs')
    target_inputs = Input(shape=(1,), name='target_inputs')
    document_inputs = Input(shape=(1,), name='document_inputs')

    word_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size,
                               embeddings_initializer=args.doc_initializer, name='word_embedding')

    # context_embedding shape [batch_size*window_size*2, embedding_size]
    context_embedding = word_embedding(context_inputs)

    document_embedding = Embedding(input_dim=num_document, output_dim=embedding_size, name='document_embedding',
                                   embeddings_initializer=args.doc_initializer)(document_inputs)

    document_embedding = Reshape(target_shape=(embedding_size,))(document_embedding)

    # mean of word embedding vector
    mean_context_embedding = Lambda(lambda x: K.mean(x, axis=1))(context_embedding)

    average_embedding = Average(name='document_vector')([document_embedding, mean_context_embedding])

    # Keras does not exists NCE-loss, so implementation NCE loss.
    nce_logits = NceLogit(target=target_inputs, vocab_size=vocab_size, num_true=num_true,
                          embedding_size=embedding_size, num_sampled=args.negative_sample)(average_embedding)
    model = Model([context_inputs, target_inputs, document_inputs], nce_logits)

    if summary:
        model.summary()

    model.compile(loss=[nce_loss], optimizer=Adam(lr=args.doc_lr))
    return model