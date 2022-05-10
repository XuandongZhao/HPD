CUDA_VISIBLE_DEVICES=2 \
python teacher_emb.py --teacher 'simcse' --final-dim 128 --batch-size 256 --save-dir './embs/'