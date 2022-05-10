## Homomorphic projective distillation (HPD) for sentence embedding

This repository contains the code and pre-trained models for our paper [Compressing Sentence Representation for Semantic Retrieval via Homomorphic Projective Distillation](https://arxiv.org/abs/2203.07687).

* Thanks for your interest in our repo!


## Overview

We propose **H**omomorphic **P**rojective **D**istillation (HPD) to learn compressed sentence embeddings. Our method augments a small Transformer encoder model with learnable projection layers to produce compact representations while mimicking a large pre-trained language model to retain the sentence representation quality. The following figure is an illustration of our models.

<div style="text-align: center"><img src="figure/model.pdf" width="400"></div>


## Getting Started

Our implementation is based on [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers). You can get an easy-to-use sentence embedding tool by installing `sentence-transformers`:

```bash
# install with pip
pip install -U sentence-transformers
# or install with conda
conda install -c conda-forge sentence-transformers
```

Note that if you want to enable GPU encoding, you should install the correct version of PyTorch that supports CUDA. See [PyTorch official website](https://pytorch.org) for instructions.

After installing the package, you can simply load our model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Xuandong/HPD-MiniLM-F128')
```

There are two available models: [`HPD-MiniLM-F128`](https://huggingface.co/Xuandong/HPD-MiniLM-F128) and [`HPD-TinyBERT-F128`](https://huggingface.co/Xuandong/HPD-TinyBERT-F128).

Then you can use our model for **encoding sentences into embeddings**
```python
sentences = ['He plays guitar.', 'A street vendor is outside.', 'A woman is reading.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```

## Pre-Trained Models

Our released models are listed as following. You can import these models by using the `sentence-transformers` package or using [HuggingFace's Transformers](https://github.com/huggingface/transformers). 

| Models               | STS-B   | Avg. STS  | Model Size   | Emb Dim  |
| -------------------- | ------- | --------- | ------------ | -------- |
| SBERT-Base           | 77.03   | 74.89     | 109M         | 768      |
| SimCSE-RoBERTa-Large | 86.70   | 83.76     | 355M         | 1024     |
| [HPD-TinyBERT-F128](https://huggingface.co/Xuandong/HPD-TinyBERT-F128)     | 84.36   | 81.02     | 14M          | 128      |
| [HPD-MiniLM-F128](https://huggingface.co/Xuandong/HPD-MiniLM-F128)      | 84.98   | 81.80     | 23M          | 128      |

## Training

You should first get the augmented data and put it in the `datasets` folder. Or you can just use ALLNLI dataset (worse performance).

By running the script, we store all teacher's embeddings to `embs/simcse-train-F128.pt` and `embs/simcse-valid-F128.pt`.
```bash
bash run_emb.sh
```

Then we can train the model. Example in script,

```bash
bash run.sh
```


## Evaluation

Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE). It evaluates sentence embeddings on semantic textual similarity (STS) tasks.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,

```bash
python senteval.py \
    --pooler avg 
    --task_set sts 
    --mode test 
    --cuda cuda:0 
    --model_type st 
    --model_name_or_path Xuandong/HPD-MiniLM-F128
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 74.94 | 84.52 | 80.25 | 84.87 | 81.90 |    84.98     |      81.15      | 81.80 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Please see [Evaluation](https://github.com/princeton-nlp/SimCSE#evaluation) of SimCSE for more details about the arguments. We add one argument to choose `sentence-transformers` model or `simcse` model.

* `--model_type`: Which model type to use. Now we support
  * `st` (default): `sentence-transformers` based models
  * `simcse`: SimCSE based models


## Citation

Please cite our paper if you use HPD in your work:

```bibtex
@article{zhao2022compressing,
  title={Compressing Sentence Representation for Semantic Retrieval via Homomorphic Projective Distillation},
  author={Zhao, Xuandong and Yu, Zhiguo and Wu, Ming and Li, Lei},
  journal={arXiv preprint arXiv:2203.07687},
  year={2022}
}
```