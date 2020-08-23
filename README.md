# BERT(S) for Relation Extraction

## Overview
A PyTorch implementation of the models for the paper ["Matching the Blanks: Distributional Similarity for Relation Learning"](https://arxiv.org/pdf/1906.03158.pdf) published in ACL 2019.  

Note: This is not an official repo for the paper.  

## Training by matching the blanks (BERT<sub>EM</sub> + MTB)
Run `pretraining.py` with a YAML `--conf_file` containing the following arguments:

```yaml
# Data
data: data/cnn.txt # pre-training data.txt file path
normalization: # How to normalize the pretraining corpus
  - lowercase # Apply lowercase
  - html # Strip HTML tags
  - urls # Remove URLs
# Model
transformer: bert-base-uncased # weight initialization (Should be huggingface BERT model)
# Training
batch_size: 32 # Training batch size
max_norm: 1.0 # Clipped gradient norm
epochs: 18 # Number of Epochs
lr: 0.0001 # learning rate
resume: False # Use this to resume the train job
```

Pre-training data can be any .txt continuous text file.  
We use Spacy NLP to grab pairwise entities (within a window size of 40 tokens length) from the text to form relation statements for pre-training. Entities recognition are based on NER and dependency tree parsing of objects/subjects.  

## Fine-tuning on SemEval2010 Task 8 (BERT<sub>EM</sub>/BERT<sub>EM</sub> + MTB)
Run `main_task.py` with a YAML `--conf_file` containing the arguments below. Requires SemEval2010 Task 8 dataset, available [here.](https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip) Download & unzip to `data/sem_eval` folder.

```yaml
# Data
train_file: data/sem_eval/SemEval2010_task8_training/TRAIN_FILE.TXT
test_file: data/sem_eval/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
# Model
pretrained_mtb_model: models/pretraining/pretraining-small/bert-base-uncased/best_model.pth.tar
transformer: bert-base-uncased
# Training
batch_size: 64
max_norm: 1.0
epochs: 25
lr: 0.00007
resume: False

```
`pretrained_mtb_model` can be `None` to use pretrained BERT coming from the `transformers` package.
