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
git clone https://github.com/Jester6136/ExternalContextNER.git
cd ExternalContextNER
pip install -r requirements.txt
```

## Import data 

```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1H0II0wMlcQ7chw4fdg95CKa07O6BC2U5 -O tmp_data
```



## Changing Labels in ENV:
   #### For vlsp2016
   ```
   export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,E,X,<s>,</s>"
```
   #### For vlsp2018
   ```
   export LABELS="I-ORGANIZATION,B-ORGANIZATION,I-LOCATION,B-MISCELLANEOUS,I-PERSON,O,B-PERSON,I-MISCELLANEOUS,B-LOCATION,E,X,<s>,</s>"
   ```

   #### For vlsp2021
   ```
   export LABELS="I-PRODUCT-AWARD,B-MISCELLANEOUS,B-QUANTITY-NUM,B-ORGANIZATION-SPORTS,B-DATETIME,I-ADDRESS,I-PERSON,I-EVENT-SPORT,B-ADDRESS,B-EVENT-NATURAL,I-LOCATION-GPE,B-EVENT-GAMESHOW,B-DATETIME-TIMERANGE,I-QUANTITY-NUM,I-QUANTITY-AGE,B-EVENT-CUL,I-QUANTITY-TEM,I-PRODUCT-LEGAL,I-LOCATION-STRUC,I-ORGANIZATION,B-PHONENUMBER,B-IP,O,B-QUANTITY-AGE,I-DATETIME-TIME,I-DATETIME,B-ORGANIZATION-MED,B-DATETIME-SET,I-EVENT-CUL,B-QUANTITY-DIM,I-QUANTITY-DIM,B-EVENT,B-DATETIME-DATERANGE,I-EVENT-GAMESHOW,B-PRODUCT-AWARD,B-LOCATION-STRUC,B-LOCATION,B-PRODUCT,I-MISCELLANEOUS,B-SKILL,I-QUANTITY-ORD,I-ORGANIZATION-STOCK,I-LOCATION-GEO,B-PERSON,B-PRODUCT-COM,B-PRODUCT-LEGAL,I-LOCATION,B-QUANTITY-TEM,I-PRODUCT,B-QUANTITY-CUR,I-QUANTITY-CUR,B-LOCATION-GPE,I-PHONENUMBER,I-ORGANIZATION-MED,I-EVENT-NATURAL,I-EMAIL,B-ORGANIZATION,B-URL,I-DATETIME-TIMERANGE,I-QUANTITY,I-IP,B-EVENT-SPORT,B-PERSONTYPE,B-QUANTITY-PER,I-QUANTITY-PER,I-PRODUCT-COM,I-DATETIME-DURATION,B-LOCATION-GPE-GEO,B-QUANTITY-ORD,I-EVENT,B-DATETIME-TIME,B-QUANTITY,I-DATETIME-SET,I-LOCATION-GPE-GEO,B-ORGANIZATION-STOCK,I-ORGANIZATION-SPORTS,I-SKILL,I-URL,B-DATETIME-DURATION,I-DATETIME-DATE,I-PERSONTYPE,B-DATETIME-DATE,I-DATETIME-DATERANGE,B-LOCATION-GEO,B-EMAIL,E,X,<s>,</s>"
```

## Usage

### Training BERT-CRF with External Context

You can train the BERT-CRF model with external context using the following command:

```bash

export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,E,X,<s>,</s>"
python train_bert_crf_EC_new_roberta.py \
    --do_train \
    --do_eval \
    --output_dir "./VLSP2016_img" \
    --bert_model "vinai/phobert-base-v2" \
    --learning_rate 3e-5 \
    --data_dir "tmp_data/VLSP2016" \
    --num_train_epochs 12 \
    --train_batch_size 36 \
    --task_name "sonba" \
    --cache_dir "cache" \
    --max_seq_length 256
```

also, I wrote some example sh file for running.