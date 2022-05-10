import os
import gzip
import csv
import random
import pickle
from sentence_transformers import util


def get_data(data_dir='datasets/AllNLI.tsv.gz', rand_seed=4, aug_data=None, valid_num=5000):
    # Download datasets if needed
    if not os.path.exists(data_dir):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', data_dir)

    train_sentences_nli = set()
    valid_sentences_nli = set()

    # Read ALLNLI
    with gzip.open(data_dir, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                valid_sentences_nli.add(row['sentence1'])
                valid_sentences_nli.add(row['sentence2'])
            else:
                train_sentences_nli.add(row['sentence1'])
                train_sentences_nli.add(row['sentence2'])

    if aug_data is not None:
        # Add aug data path
        with open('./datasets/aug_wordnet_sub_1.pkl', 'rb') as f:
            aug_list_1_1 = pickle.load(f)
        with open('./datasets/aug_back_trans_1.pkl', 'rb') as f:
            aug_list_1_2 = pickle.load(f)

        with open('./datasets/aug_wordnet_sub_2.pkl', 'rb') as f:
            aug_list_2_1 = pickle.load(f)
        with open('./datasets/aug_back_trans_2.pkl', 'rb') as f:
            aug_list_2_2 = pickle.load(f)

        aug_list = aug_list_1_1 + aug_list_1_2 + aug_list_2_1 + aug_list_2_2
        aug_list = set(aug_list)

        train_sentences_nli = sorted(list(train_sentences_nli) + list(aug_list))
    else:
        train_sentences_nli = sorted(list(train_sentences_nli))

    print('First 3 sentences for train:\n', train_sentences_nli[0:3])

    valid_sentences_nli = sorted(list(valid_sentences_nli))
    print('First 3 sentences for valid:\n', valid_sentences_nli[0:3])

    random.Random(rand_seed).shuffle(train_sentences_nli)
    print('Train after randomization:', len(train_sentences_nli))
    print(train_sentences_nli[0:3])

    random.Random(rand_seed).shuffle(valid_sentences_nli)
    print('Valid after randomization:', len(valid_sentences_nli))
    print(valid_sentences_nli[0:3])

    valid_sentences_nli = valid_sentences_nli[0:valid_num]
    return train_sentences_nli, valid_sentences_nli
