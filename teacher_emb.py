import numpy as np
import torch
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from simcse import SimCSE
from data import get_data


def get_teacher_emb(teacher='simcse', final_dim=128, batch_size=256, save_dir='./embs/'):
    train_sentences, valid_sentences = get_data()

    # You can also get other state-of-the-art sentence embeddings by changing the teacher model
    if teacher == 'simcse':
        model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
        teacher_dim = 1024
        train_embeddings = model.encode(train_sentences, batch_size=batch_size)
        valid_embeddings = model.encode(valid_sentences, batch_size=batch_size)
    elif teacher == 'st':
        # model = SentenceTransformer('stsb-roberta-base-v2')
        # teacher_dim = 1024
        # model = SentenceTransformer('stsb-mpnet-base-v2')
        # teacher_dim = 768
        model = SentenceTransformer('nli-mpnet-base-v2')
        teacher_dim = 768
        train_embeddings = torch.tensor(model.encode(train_sentences, batch_size=batch_size))
        valid_embeddings = torch.tensor(model.encode(valid_sentences, batch_size=batch_size))
    else:
        raise ValueError("No Teacher Model available")

    print(train_embeddings.shape)
    print(valid_embeddings.shape)

    pca = PCA(n_components=final_dim)
    pca.fit(train_embeddings[0:40000])
    pca_comp = np.asarray(pca.components_)

    dense = torch.nn.Linear(teacher_dim, final_dim, bias=False)
    dense.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    train_file_path = save_dir + teacher + '-train-F' + str(final_dim) + '.pt'
    valid_file_path = save_dir + teacher + '-valid-F' + str(final_dim) + '.pt'
    torch.save(dense(train_embeddings.double()), train_file_path)
    torch.save(dense(valid_embeddings.double()), valid_file_path)

    print('Finish teacher embedding, save to', train_file_path, valid_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for getting teacher's embeddings")
    parser.add_argument("--teacher", type=str, default='simcse', choices=['simcse', 'st'], help='teacher model')
    parser.add_argument("--final-dim", type=int, default=128, help="final dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--save-dir", type=str, default='./embs/', help="save path")
    args = parser.parse_args()
    get_teacher_emb(teacher=args.teacher, final_dim=args.final_dim, batch_size=args.batch_size, save_dir=args.save_dir)
