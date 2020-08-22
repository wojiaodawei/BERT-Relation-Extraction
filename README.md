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

### Inference (--infer=1)
To infer a sentence, you can annotate entity1 & entity2 of interest within the sentence with their respective entities tags [E1], [E2]. 
Example:
```bash
Type input sentence ('quit' or 'exit' to terminate):
The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.

Sentence:  The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.
Predicted:  Cause-Effect(e1,e2) 
```

```python
from src.tasks.infer import infer_from_trained

inferer = infer_from_trained(args, detect_entities=False)
test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
inferer.infer_sentence(test, detect_entities=False)
```
```bash
Sentence:  The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.
Predicted:  Cause-Effect(e1,e2) 
```

The script can also automatically detect potential entities in an input sentence, in which case all possible relation combinations are inferred:
```python
inferer = infer_from_trained(args, detect_entities=True)
test2 = "After eating the chicken, he developed a sore throat the next morning."
inferer.infer_sentence(test2, detect_entities=True)
```
```bash
Sentence:  [E2]After eating the chicken[/E2] , [E1]he[/E1] developed a sore throat the next morning .
Predicted:  Other 

Sentence:  After eating the chicken , [E1]he[/E1] developed [E2]a sore throat[/E2] the next morning .
Predicted:  Other 

Sentence:  [E1]After eating the chicken[/E1] , [E2]he[/E2] developed a sore throat the next morning .
Predicted:  Other 

Sentence:  [E1]After eating the chicken[/E1] , he developed [E2]a sore throat[/E2] the next morning .
Predicted:  Other 

Sentence:  After eating the chicken , [E2]he[/E2] developed [E1]a sore throat[/E1] the next morning .
Predicted:  Other 

Sentence:  [E2]After eating the chicken[/E2] , he developed [E1]a sore throat[/E1] the next morning .
Predicted:  Cause-Effect(e2,e1) 
```

## FewRel Task
Download the FewRel 1.0 dataset [here.](https://drive.google.com/drive/folders/1ljobnuzxStFQJSlN4ZHMcMhZtEYaRAHy?usp=sharing) and unzip to ./data/ folder.  
Run main_task.py with argument 'task' set as 'fewrel'.
```bash
python main_task.py --task fewrel
```
Results:  
(5-way 1-shot)  
BERT<sub>EM</sub> without MTB, not trained on any FewRel data  
| Model size | Accuracy (41646 samples) |
|------------|--------------------------|
| bert-base-uncased  | 62.229 %         |
| bert-large-uncased | 72.766 %         |


## Benchmark Results

### SemEval2010 Task 8
1) Base architecture: BERT base uncased (12-layer, 768-hidden, 12-heads, 110M parameters)

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_0.png) 


2) Base architecture: ALBERT base uncased (12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters)  

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_1.png) 

## To add
- inference & results on benchmarks (SemEval2010 Task 8) with MTB pre-training
- felrel task

