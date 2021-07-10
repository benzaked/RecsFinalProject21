import numpy as np
import pandas as pd
import os.path
from os import path
import tensorflow as tf
from argparse import ArgumentParser
from batch import get_batch_test_data
from evaluate import evaluate
import heapq

from NAIS import NAIS
from dataset import DataSet

def parse_args():
    parser = ArgumentParser(description='Run NAIS.')
    parser.add_argument('--path', nargs='?', default='data',
                        help='Input data path.')
    parser.add_argument('--data_set_name', nargs='?', default='ml-1m',
                        help='Choose a dataset, either ml-1m or pinterest-20.')
    parser.add_argument('--topN', type=int, default=20,
                        help='Size of recommendation list.')
    parser.add_argument('--checkpoint_name', nargs='?', default='NAIS_1624541449',
                        help='Size of recommendation list.')
    parser.add_argument('--save_pred_path', nargs='?', default='predictions/NAIS/ml-1m/16-prod',
                        help='Size of recommendation list.')


    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='whether pretraining or not, 1-pretrain, 0-without pretrain.')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--attention_factor', type=int, default=16,
                        help='Attention factor.')
    parser.add_argument('--algorithm', type=str, default='prod',
                        help='Either concat or prod')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Smoothing exponent of softmax.')
    parser.add_argument('--regs', nargs='?', default='(1e-7, 1e-7, 1e-5, 1e-7, 1e-7)',
                        help='Regularization parameter.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per iteration.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    topN = args.topN
    dataset = DataSet(path=args.path,
                          data_set_name=args.data_set_name)

    nais = NAIS(num_users=dataset.num_users,
                num_items=dataset.num_items,
                args=args)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=args.lr,
                                            initial_accumulator_value=1e-8)
    checkpoint = tf.train.Checkpoint(model=nais,
                                     optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=f'pretrain/NAIS/{args.data_set_name}',
                                         checkpoint_name=f'{args.checkpoint_name}.ckpt',
                                         max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)
    hits, ndcgs, mrrs = [], [], []
    test_rank_list_data = []
    for batch_id in range(dataset.num_users):
        user_input, item_input, test_item, n_u = get_batch_test_data(batch_id=batch_id,
                                                                     dataset=dataset)
        predictions = nais.predict(user_input=user_input,
                                    item_input=item_input,
                                    num_idx=n_u)
        map_item_score = {}
        for i in range(len(item_input)):
            item = item_input[i]
            map_item_score[item] = predictions[i]

        rank_list = heapq.nlargest(topN, map_item_score, key=map_item_score.get)
        test_rank_list_data.append([batch_id, test_item, rank_list])
        hit, ndcg, mrr = evaluate(rank_list, test_item)
        hits.append(hit)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    test_hr, test_ndcg, test_mrr = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(mrrs).mean()
    print(f'data_set_name:{args.data_set_name}, model: {args.checkpoint_name}, HR@{topN}: {test_hr}, NDCG@{topN}: {test_ndcg}, MRR@{topN}: {test_mrr}')
    test_rank_list_data_df = pd.DataFrame(test_rank_list_data, columns=['user', 'ranked_item', 'predicted_list'])
    if not path.exists(args.save_pred_path):
        os.makedirs(args.save_pred_path)
    test_rank_list_data_df.to_csv(f'{args.save_pred_path}/predictions.csv', index=False)
