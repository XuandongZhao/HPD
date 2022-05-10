import torch
import argparse
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, SentenceTransformer, models, losses, evaluation
from data import get_data
from dataset import SentencesDataset
from evaluator import MSEEval


def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, handlers=[LoggingHandler()])
    epoch_num = args.epochs
    train_batch_size = args.train_batch_size
    infer_batch_size = args.infer_batch_size
    final_dim = args.final_dim
    output_path = args.output_dir + 'HPD-F' + str(final_dim) + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    teacher_train_emb = torch.load(args.teacher_train_emb)
    teacher_valid_emb = torch.load(args.teacher_valid_emb)
    student_word_embed_model = models.Transformer(args.student_base)
    student_pool_model = models.Pooling(student_word_embed_model.get_word_embedding_dimension())
    student_proj_model = models.Dense(in_features=student_pool_model.get_sentence_embedding_dimension(),
                                      out_features=args.final_dim, activation_function=torch.nn.Tanh())
    student_model = SentenceTransformer(modules=[student_word_embed_model, student_pool_model, student_proj_model])
    train_sentences, valid_sentences = get_data(args.data_dir)

    train_data = SentencesDataset(student_model=student_model, teacher_model=None,
                                  batch_size=infer_batch_size, use_embedding_cache=True)
    train_data.add_dataset([[sent] for sent in train_sentences], max_sentence_length=256)
    train_data.add_emb_cache(train_sentences, teacher_train_emb)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MSELoss(model=student_model)
    valid_evaluator = MSEEval(teacher_valid_emb, valid_sentences)

    # Train the student model to imitate the teacher
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluation.SequentialEvaluator([valid_evaluator]),
                      epochs=epoch_num,
                      warmup_steps=args.warmup_steps,
                      evaluation_steps=args.evaluation_steps,
                      output_path=output_path,
                      save_best_model=True,
                      optimizer_params={'lr': args.lr, 'eps': 1e-6, 'correct_bias': False},
                      use_amp=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="warmup steps for the training")
    parser.add_argument("--evaluation-steps", type=int, default=5000, help="evaluation steps for the training")
    parser.add_argument("--train-batch-size", type=int, default=64, help="batch size for training")
    parser.add_argument("--infer-batch-size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--final-dim", type=int, default=128, help="final dimension")
    parser.add_argument("--output-dir", type=str, default='./output/', help="path to save model")
    parser.add_argument("--data-dir", type=str, default='./datasets/AllNLI.tsv.gz', help="path to training data")
    parser.add_argument("--teacher-train-emb", type=str, help="path to teacher's embedding for training")
    parser.add_argument("--teacher-valid-emb", type=str, help="path to teacher's embedding for validation")
    parser.add_argument("--student-base", type=str, default='tinybert-L4-H312', help='student model')

    args = parser.parse_args()

    main(args)
