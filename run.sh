CUDA_VISIBLE_DEVICES=1 \
python train.py \
--epochs 5 --lr 1e-4 --train-batch-size 256 --infer-batch-size 256 \
--final-dim 128 --output-dir './output/' --data-dir './datasets/AllNLI.tsv.gz' \
--teacher-train-emb './embs/simcse-train-F128.pt' \
--teacher-valid-emb './embs/simcse-valid-F128.pt' \
--student-base 'nreimers/MiniLM-L6-H384-uncased'
# nreimers/MiniLM-L6-H384-uncased