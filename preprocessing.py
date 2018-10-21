import pandas as pd
import os
import numpy as np

from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer

from keras.preprocessing.text import Tokenizer


def load_dataset(args):

    df = pd.read_csv(os.path.join(args.dataset_path, "kpris_data.csv"))

    # label encoder
    le = preprocessing.LabelEncoder()

    # categories select
    if args.dataset != "5_categories":

        categories = args.dataset.split("_")
        target_index = np.where(np.isin(df.target.values, categories) == 1)[0]

        x_data = df.abstract.values[target_index]
        y_data = le.fit_transform(df.target.values[target_index])

    # use all categories of KPRIS dataset
    else:
        x_data = df.abstract.values
        y_data = le.fit_transform(df.target.values)

    assert len(x_data) == len(y_data)
    print("Number of Abstract : {} , Target : {}".format(len(x_data), len(y_data)))

    return x_data, y_data


def stemming(sentences):

    stemmer = SnowballStemmer("english")

    stemming_sentences = []
    for i, sent in enumerate(sentences):
        stem_sent = " ".join([stemmer.stem(word) for word in sent.split()])
        stemming_sentences.append(stem_sent)

    print("Stemming process done")
    return stemming_sentences


def get_sequences(sentences, args):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    # get word index
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # get vocab size, use index zero to padding
    vocab_size = len(word_index) + 1
    print("Vocab size is {}".format(vocab_size))

    # padding sequence same as window size.
    sequences = [[0]*args.window_size+sequence+[0]*args.window_size for sequence in sequences]

    # check how many training sampling we get
    instances = np.sum([len(sequence)-2*args.window_size for sequence in sequences])
    print("Training sampling : {}".format(instances))

    return sequences, word_index, vocab_size, instances


def get_trainable_data(sequences, instances, args):

    context = np.zeros(shape=(instances, args.window_size*2+1), dtype=np.int32)
    target = np.zeros(shape=(instances, 1), dtype=np.int32)
    document = np.zeros(shape=(instances, 1), dtype=np.int32)

    k = 0
    for doc_id, sequence in enumerate(sequences):
        for i in range(args.window_size, len(sequence)-args.window_size):

            context[k] = sequence[i-args.window_size:i+args.window_size+1]
            target[k] = sequence[i]
            document[k] = doc_id
            k += 1

    # delete target word in context
    context = np.delete(context, args.window_size, axis=1)

    print("trainable data settting finish")
    return context, target, document

