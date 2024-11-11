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
├── run_vlsp2016_more_image_roberta.sh        # Script to run Twitter dataset
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

## Usage

### Training BERT-CRF with External Context

You can train the BERT-CRF model with external context using the following command:

```bash
sh run_vlsp2016_more_image_roberta.sh
```
