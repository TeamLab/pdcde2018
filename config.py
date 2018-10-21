import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices='car_camera memory_cpu 5_categories'.split(),
                        help='select categories "car_camera, memory_cpu, 3_categories, 5_categories"')

    parser.add_argument('--gpu_id',
                        type=str,
                        default="0",
                        help='select gpu id. default setting is "0"'
                        )

    parser.add_argument('--save_embedding_vector',
                        default='./embedding_vector',
                        type=str,
                        help='save path of patent embedding vectors'
                        )

    parser.add_argument('--save_log_path',
                        default='./train_log',
                        type=str,
                        help='save path of train log csv file'
                        )

    parser.add_argument('--save_weight_path',
                        default='./checkpoint',
                        type=str,
                        help='save weights path'
                        )

    parser.add_argument('--dataset_path',
                        default='./dataset',
                        type=str,
                        help='KPRIS and KISTA dataset path')

    parser.add_argument('--window_size',
                        default=5,
                        type=int,
                        help='doc2vec window size. default is 5.')

    parser.add_argument('--embedding_size',
                        default=50,
                        type=int,
                        help='embedding vector dimension')

    parser.add_argument('--doc_initializer',
                        default='uniform',
                        type=str,
                        help='Doc2Vec word and document initializer'
                        )

    parser.add_argument('--negative_sample',
                        default=5,
                        type=int,
                        help='number of negative sampling used nce loss.')

    parser.add_argument('--doc_lr',
                        default=0.001,
                        type=float,
                        help='Doc2Vec initial learning rate')

    parser.add_argument('--doc_batch_size',
                        default=256,
                        type=int,
                        help='Doc2Vec batch size')

    parser.add_argument('--doc_epochs',
                        default=500,
                        type=int,
                        help='Doc2Vec epochs')

    parser.add_argument('--doc_decay_step',
                        default=50,
                        type=float,
                        help='decay step. Default 0.5 decay every 200 epochs')

    parser.add_argument('--dec_batch_size',
                        default=256,
                        type=int,
                        help='deep cluster embedding model batch size')

    parser.add_argument('--dec_lr',
                        default=0.001,
                        type=float,
                        help='deep cluster embedding model initial learning rate')

    parser.add_argument('--dec_decay_step',
                        default=20,
                        type=int,
                        help='deep cluster embedding model learning rate decay')

    parser.add_argument('--task',
                        default='train',
                        type=str,
                        help='select train test')

    parser.add_argument('--layerwise_pretrain_iters',
                        default=10000,
                        type=int,
                        help='layer-wise pretrain weight for greedy layer wise auto encoder')

    parser.add_argument('--finetune_iters',
                        default=20000,
                        type=int,
                        help='fine-tunning iteration')

    return parser.parse_args()