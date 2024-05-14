import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,BertConfig
from modules.model_architecture.bert_crf_EC_new_xlm_roberta import XLMRoberta_token_classification
from modules.datasets.dataset_bert_EC_new_xlm_roberta import convert_mm_examples_to_features,MNERProcessor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam,warmup_linear
from ner_evaluate import evaluate_each_class
from seqeval.metrics import classification_report
from ner_evaluate import evaluate
from tqdm import tqdm, trange
import json
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--alpha",
                    default=0.5,
                    type=float,
                    help="parameter for Conversion Matrix")

parser.add_argument("--beta",
                    default=0.5,
                    type=float,
                    help="parameter for aux loss")


parser.add_argument("--sigma",
                    default=0.005,
                    type=float,
                    help="parameter for PixelCNN loss")


parser.add_argument("--theta",
                    default=0.05,
                    type=float,
                    help="parameter for CL loss")


parser.add_argument("--weight_decay_pixelcnn",
                    default=0.0,
                    type=float,
                    help="weight_decay for PixelCNN++")

parser.add_argument("--lr_pixelcnn",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for PixelCNN++")

parser.add_argument('--lamb',
                    default=0.62,
                    type=float)

parser.add_argument('--temp',
                    type=float,
                    default=0.179,
                    help="parameter for CL training")

parser.add_argument('--temp_lamb',
                    type=float,
                    default=0.7,
                    help="parameter for CL training")

parser.add_argument("--data_dir",
                    default='./data/twitter2017',
                    type=str,

                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default='bert-base-cased', type=str)
parser.add_argument("--task_name",
                    default='twitter2017',
                    type=str,

                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./output_result',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--cache_dir",
                    default="",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--train_batch_size",
                    default=64,
                    type=int,
                    help="Total batch size for training.")

parser.add_argument("--eval_batch_size",
                    default=16,
                    type=int,
                    help="Total batch size for eval.")

parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--num_train_epochs",
                    default=12.0,
                    type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")

parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")

parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")

parser.add_argument('--seed',
                    type=int,
                    default=37,
                    help="random seed for initialization")

parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")

parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")

parser.add_argument('--mm_model', default='MTCCMBert', help='model name')  # 'MTCCMBert', 'NMMTCCMBert'
parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layer')
parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layer')
parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layer')
parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
parser.add_argument('--resnet_root', default='./out_res', help='path the pre-trained cnn models')
parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
parser.add_argument('--path_image', default='./IJCAI2019_data/twitter2017_images/', help='path to images')
# parser.add_argument('--mm_model', default='TomBert', help='model name') #
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
args = parser.parse_args()

if args.task_name == "twitter2017":
    args.path_image = "./IJCAI2019_data/twitter2017_images/"
elif args.task_name == "twitter2015":
    args.path_image = "./IJCAI2019_data/twitter2015_images/"

if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

processors = {
    "twitter2015": MNERProcessor,
    "twitter2017": MNERProcessor,
    "sonba": MNERProcessor
}

if args.local_rank == -1 or args.no_cuda:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

# '''
#
if args.task_name == "twitter2015":
    args.num_train_epochs = 24.0
if args.task_name == "twitter2017":
    args.num_train_epochs = 23.0
# '''

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

task_name = args.task_name.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
label_list = processor.get_labels()
auxlabel_list = processor.get_auxlabels()
num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

start_label_id = processor.get_start_label_id()
stop_label_id = processor.get_stop_label_id()

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

if args.mm_model == 'MTCCMBert':
    model = XLMRoberta_token_classification.from_pretrained(args.bert_model,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels_=num_labels)
else:
    print('please define your MNER Model')

# net = getattr(resnet, 'resnet152')()
# net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
# encoder = myResnet(net, args.fine_tune_cnn, device)

if args.fp16:
    model.half()
model.to(device)
if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)




else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")


temp = args.temp
temp_lamb = args.temp_lamb
lamb = args.lamb


if args.do_train:
    train_features = convert_mm_examples_to_features(
        train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.long)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.long)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_label_ids2 = torch.tensor([f.label_id2 for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_input_ids2, all_input_mask2,all_segment_ids2, all_label_ids,all_label_ids2)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    dev_eval_examples = processor.get_dev_examples(args.data_dir)
    dev_eval_features = convert_mm_examples_to_features(
        dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in dev_eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in dev_eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in dev_eval_features], dtype=torch.long)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in dev_eval_features], dtype=torch.long)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in dev_eval_features], dtype=torch.long)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in dev_eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in dev_eval_features], dtype=torch.long)
    all_label_ids2 = torch.tensor([f.label_id2 for f in dev_eval_features], dtype=torch.long)

    dev_eval_data = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_input_ids2, all_input_mask2,all_segment_ids2, all_label_ids,all_label_ids2)
    # Run prediction for full data
    dev_eval_sampler = SequentialSampler(dev_eval_data)
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=dev_eval_sampler, batch_size=args.eval_batch_size)

    max_dev_f1 = 0.0
    best_dev_epoch = 0
    logger.info("***** Running training *****")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("********** Epoch: " + str(train_idx) + " **********")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2,label_id,label_id2 = batch

            print()
            print(input_ids.shape)
            print(segment_ids.shape)
            print(input_mask.shape)
            print(input_ids2.shape)
            print(segment_ids2.shape)
            print(input_mask2.shape)
            print(label_id.shape)
            print(label_id2.shape)

            neg_log_likelihood = model(input_ids,segment_ids,input_mask,input_ids2,segment_ids2,input_mask2,label_id,label_id2)

            if n_gpu > 1:
                neg_log_likelihood = neg_log_likelihood.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(neg_log_likelihood)
            else:
                neg_log_likelihood.backward()

            tr_loss += neg_log_likelihood.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                        args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        print(f"===============Main loss: {tr_loss/nb_tr_steps}===============")

        model.eval()

        logger.info("***** Running Dev evaluation *****")
        logger.info("  Num examples = %d", len(dev_eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "pad"
        for input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2,label_id,label_id2 in tqdm(
                dev_eval_dataloader,
                desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            input_ids2 = input_ids2.to(device)
            input_mask2 = input_mask2.to(device)
            segment_ids2 = segment_ids2.to(device)
            label_ids = label_id.to(device)
            label_ids2 = label_id2.to(device)

            with torch.no_grad():
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>" and label_map[label_ids[i][j]] != "/s>" and label_map[label_ids[i][j]] != "E":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)


        report = classification_report(y_true, y_pred, digits=4)
        sentence_list = []
        dev_data, _ = processor._read_sbtsv(os.path.join(args.data_dir, "dev.txt"))
        for i in range(len(y_pred)):
            sentence = dev_data[i][0]
            sentence_list.append(sentence)

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)

        logger.info("***** Dev Eval results *****")
        logger.info("\n%s", report)
        print("Overall: ", p, r, f1)
        F_score_dev = f1

        if F_score_dev >= max_dev_f1:
            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                            "label_map": label_map}
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
            max_dev_f1 = F_score_dev
            best_dev_epoch = train_idx

    print("**************************************************")
    print("The best epoch on the dev set: ", best_dev_epoch)
    print("The best Overall-F1 score on the dev set: ", max_dev_f1)
    print('\n')

# loadmodel
if args.mm_model == 'MTCCMBert':
    model = XLMRoberta_token_classification.from_pretrained(args.bert_model, num_labels_=num_labels)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
else:
    print('please define your MNER Model')

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_mm_examples_to_features(
        eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer)
    logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in eval_features], dtype=torch.long)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in eval_features], dtype=torch.long)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_label_ids2 = torch.tensor([f.label_id2 for f in eval_features], dtype=torch.long)
    
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_input_ids2, all_input_mask2,all_segment_ids2, all_label_ids,all_label_ids2)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    y_true_idx = []
    y_pred_idx = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "pad"
    for input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2,label_id,label_id2 in tqdm(
            eval_dataloader, desc="Evaluating"):
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        input_ids2 = input_ids2.to(device)
        input_mask2 = input_mask2.to(device)
        segment_ids2 = segment_ids2.to(device)
        label_ids = label_id.to(device)
        label_ids2 = label_id2.to(device)

        with torch.no_grad():
            predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2)

        logits = predicted_label_seq_ids
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            tmp1_idx = []
            tmp2_idx = []

            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>" and label_map[label_ids[i][j]] != "/s>" and label_map[label_ids[i][j]] != "E":
                        temp_1.append(label_map[label_ids[i][j]])
                        tmp1_idx.append(label_ids[i][j])
                        temp_2.append(label_map[logits[i][j]])
                        tmp2_idx.append(logits[i][j])
                else:

                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
            y_true_idx.append(tmp1_idx)
            y_pred_idx.append(tmp2_idx)


    report = classification_report(y_true, y_pred, digits=4)

    sentence_list = []
    test_data, _ = processor._read_sbtsv(os.path.join(args.data_dir, "test.txt"))
    output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
    fout = open(output_pred_file, 'w')
    for i in range(len(y_pred)):
        sentence = test_data[i][0]
        sentence_list.append(sentence)
        samp_pred_label = y_pred[i]
        samp_true_label = y_true[i]
        fout.write(' '.join(sentence) + '\n')
        fout.write(' '.join(samp_pred_label) + '\n')
        fout.write(' '.join(samp_true_label) + '\n' + '\n')
    fout.close()

    reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    print("Overall: ", p, r, f1)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')