import config
import preprocessing
import utils
import doc2vec
import numpy as np
import os
import pickle


if __name__ == "__main__":

    args = config.get_args()

    # make directory
    utils.make_directory_doc(args)

    # load dataset
    abstract, label = preprocessing.load_dataset(args)

    # stemming process. we used Snowball stemming of nltk package.
    abstract = preprocessing.stemming(abstract)

    # convert word text to idx.
    sequences, word2idx, vocab_size, instances = preprocessing.get_sequences(abstract, args)

    # get context words, target word and document idx
    context, target, document = preprocessing.get_trainable_data(sequences, instances, args)

    num_document = np.max(document)+1

    # model load and compile
    model = doc2vec.load_model(args, vocab_size, args.embedding_size, num_document)

    # get callbacks
    callbacks = utils.get_callbacks_doc(args)

    if not os.path.isfile(os.path.join(args.save_weight_path, "doc2vec_weights_{}.h5".format(args.dataset))):
        # train
        model.fit(x=[context, target, document], y=target, shuffle=True,
                  batch_size=args.doc_batch_size, epochs=args.doc_epochs, callbacks=callbacks)

    # save patent abstract vector
    model.load_weights(os.path.join(args.save_weight_path, "doc2vec_weights_{}.h5".format(args.dataset)))
    embedding_vector = model.get_weights()[0]

    if not os.path.isfile(os.path.join(args.save_embedding_vector, args.dataset)):
        pickle.dump([embedding_vector, label], open(os.path.join(args.save_embedding_vector, args.dataset), "wb"))

    print("Finish embedding process!")
