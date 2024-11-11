# ExternalContext - Re-Implementation: Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning

This repository contains a re-implementation of the Named Entity Recognition (NER) task, enhanced with external context retrieval and cooperative learning techniques. The project focuses on improving NER performance by integrating external context data during training and inference.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

The key files and directories are:

```
.
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── train_bert_crf.py              # Training script for BERT+CRF
├── train_bert_crf_EC_new.py        # Training with external context (new model)
├── train_bert_crf_roberta.py       # Training for Roberta
├── modules/                       # Model architectures and datasets
│   ├── model_architecture/
│   ├── datasets/
├── tmp_data/                      # Temporary data storage
├── run_vlsp2016_more_image_roberta.sh        # Script to run vlsp dataset
├── ner_evaluate.py                # Evaluation script for NER models
```

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- External libraries in `requirements.txt`

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your_username/ExternalContext.git
cd ExternalContext
pip install -r requirements.txt
```

## Import data 

```
git clone https://huggingface.co/jester6136/vlsp_external_context tmp_data
```



## Changing Labels in `dataset_bert_EC_new_roberta.py` (Sorry about this, it's my fault for being lazy.)

In the file `modules/datasets/dataset_bert_EC_new_roberta.py`, you can change the label definitions as follows:

1. Locate Label Definitions:  
   Go to line 140 - 147, where the dataset labels are defined.

2. Comment Out Unnecessary Datasets:  
   If you’re working with a specific dataset, such as `vlsp2016`, you may want to comment out labels for other datasets. For example, you can retain only the `vlsp2016` labels.

3. Example for `vlsp2016` Labels:
   To configure for `vlsp2016`, update the return line as follows:

```python
# For vlsp2016
return ["B-ORG", "B-MISC", "I-PER", "I-ORG", "B-LOC", "I-MISC", "I-LOC", "O", "B-PER", "E", "X", "<s>", "</s>"]
```

## Usage

### Training BERT-CRF with External Context

You can train the BERT-CRF model with external context using the following command:

```bash
python train_bert_crf_EC_new_roberta.py \
    --do_train \
    --do_eval \
    --output_dir "./VLSP2016_img" \
    --bert_model "vinai/phobert-base-v2" \
    --learning_rate 3e-5 \
    --data_dir "tmp_data/vlsp_MoRe_PHO_kc_image/VLSP2016" \
    --num_train_epochs 12 \
    --train_batch_size 8 \
    --task_name "sonba" \
    --cache_dir "cache" \
    --max_seq_length 256
```

also, I wrote some example sh file for 
