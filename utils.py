import os
import math
import numpy as np
from sklearn.cluster import KMeans
from keras import callbacks
from sklearn.utils.linear_assignment_ import linear_assignment


def make_directory_doc(args):

    if not os.path.exists(args.save_weight_path):
        os.makedirs(args.save_weight_path)
    if not os.path.exists(args.save_log_path):
        os.makedirs(args.save_log_path)
    if not os.path.exists(args.save_embedding_vector):
        os.makedirs(args.save_embedding_vector)


def get_callbacks_doc(args):
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.doc_lr * math.pow(0.5, math.floor((1+epoch)/args.doc_decay_step)))
    log = callbacks.CSVLogger(os.path.join(args.save_log_path, "doc2vec_log_{}.csv".format(args.dataset)))
    checkpoint = callbacks.ModelCheckpoint(os.path.join(args.save_weight_path,
                                                        "doc2vec_weights_{}.h5".format(args.dataset)),
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1,
                                           mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='loss', mode='min', patience=30)
    return [log, lr_decay, checkpoint, early_stopping]


def get_callbacks_ae(args):
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.dec_lr * math.pow(0.5, math.floor((1+epoch)/args.dec_decay_step)))
    log = callbacks.CSVLogger(os.path.join(args.save_log_path, "ae_log_{}.csv".format(args.dataset)))
    checkpoint = callbacks.ModelCheckpoint(os.path.join(args.save_weight_path,
                                                        "ae_weights_{}.h5".format(args.dataset)),
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1,
                                           mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='loss', mode='min', patience=50)
    return [log, lr_decay, checkpoint, early_stopping]


def cluster_acc(y_true, y_pred):

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    accuracy = sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w
    return accuracy[0]


def normalization_vector(embedding_vector):

    norm = np.sqrt(np.sum(np.square(embedding_vector), axis=1))
    norm = np.expand_dims(norm, axis=-1)
    normalized_doc_embeddings = embedding_vector / norm

    return normalized_doc_embeddings


def check_kmean_accuracy(normalized_doc_embeddings, label):

    kmeans = KMeans(n_clusters=len(np.unique(label)), n_init=20)
    y_pred = kmeans.fit_predict(normalized_doc_embeddings)

    accuracy = cluster_acc(label, y_pred)
    print("K-means Accuracy of Doc2Vec : {}".format(accuracy))