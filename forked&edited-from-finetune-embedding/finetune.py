# Fine tuning

# In this notebook, we finetune an opensource sentencetransformers embedding model on our synthetically generated dataset.

### Load pretrained model

from sentence_transformers import SentenceTransformer

model_id = "BAAI/bge-large-en-v1.5"
model = SentenceTransformer(model_id)

### Define dataloader

import json

from torch.utils.data import DataLoader
from sentence_transformers import InputExample

DATA_PATH = "/home/75y/data_ragMimic/data/"

TRAIN_DATASET_FPATH = DATA_PATH+'train_dataset.json'
VAL_DATASET_FPATH = DATA_PATH+'val_dataset.json'

# We use a very small batchsize to run this toy example on a local machine. 
# This should typically be much larger. 
BATCH_SIZE = 16

with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)

dataset = train_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(
    examples, batch_size=BATCH_SIZE
)

### Define loss

# **MultipleNegativesRankingLoss** is a great loss function if you only have positive pairs, for example, only pairs of similar texts like pairs of paraphrases, pairs of duplicate questions, pairs of (query, response), or pairs of (source_language, target_language).

# This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc)) as it will sample in each batch n-1 negative docs randomly.

# The performance usually increases with increasing batch sizes.

# For more detals, see:
# * [docs](https://www.sbert.net/docs/package_reference/losses.html)

from sentence_transformers import losses

loss = losses.MultipleNegativesRankingLoss(model)

### Define evaluator 

# We setup an evaluator with our val split of the dataset to monitor how well the embedding model is performing during training.

from sentence_transformers.evaluation import InformationRetrievalEvaluator

corpus_val = val_dataset['corpus']
queries_val = val_dataset['queries']
relevant_docs_val = val_dataset['relevant_docs']

evaluator = InformationRetrievalEvaluator(queries_val, corpus_val, relevant_docs_val)

### Run training 

# The training loop is very straight forward to steup thanks to sentencetransformers' high-level model training API.
# All we need to do is plugging in the data loader, loss function, and evaluator that we defined in the previous cells (along with a couple of additional minor settings).

# We train the model for very few epochs in this toy example.
# This should typically be higher for better performance.
EPOCHS = 10

warmup_steps = int(len(loader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='exp_finetune',
    show_progress_bar=True,
    evaluator=evaluator, 
    evaluation_steps=1000,
)

