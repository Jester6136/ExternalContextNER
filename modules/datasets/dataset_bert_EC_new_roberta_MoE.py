import torch 
import json
import logging
import os
from PIL import Image
import requests
import gc
import torch
logger = logging.getLogger(__name__)
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
URL_PREFIX = 'http'

class SBInputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,guid,text_a,text_b, img_id,label=None,auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a=text_a
        self.text_b=text_b
        self.label = label
        self.img_id = img_id
        # Please note that the auxlabel is not used in SB
        # it is just kept in order not to modify the original code
        self.auxlabel = auxlabel

class SBInputExampleText(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,guid,text_a,text_b,label=None,auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a=text_a
        self.text_b=text_b
        self.label = label
        # Please note that the auxlabel is not used in SB
        # it is just kept in order not to modify the original code
        self.auxlabel = auxlabel
class SBInputFeatures(object):
    """A single set of features of data"""

    def __init__(self,input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2, img_feat,label_id,label_id2,auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.img_feat = img_feat
        self.label_id = label_id
        self.label_id2 = label_id2
        self.auxlabel_id = auxlabel_id

class SBInputFeaturesText(object):
    """A single set of features of data"""

    def __init__(self,input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2,label_id,label_id2,auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.label_id = label_id
        self.label_id2 = label_id2
        self.auxlabel_id = auxlabel_id

def sbreadfile(filename,do_lower=False):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    print("prepare data for ",filename)
    f = open(filename,encoding='utf8')
    data = []
    auxlabels = []
    sentence = []
    imgs = []
    label = []
    auxlabel = []
    a = 0
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1] + '.jpg'
            continue

        if line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                if imgid!='': imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                auxlabel = []
            continue
        splits = line.split('\t')
        if do_lower:
            splits[0] = splits[0].lower()

        if splits[0] == "<eos>":
            splits[0] = "</s>"
        if splits[0] == "<EOS>":
            splits[0] = "</s>"
        if splits[0] == '' or splits[0].isspace() or splits[0] in SPECIAL_TOKENS or splits[0].startswith(URL_PREFIX):
            splits[0] = "<unk>"
        
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])
    if len(sentence) > 0:
        data.append((sentence, label))
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: " + str(len(data)))
    print("The number of images: " + str(len(imgs)))
    return data, imgs, auxlabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_sbtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return sbreadfile(input_file)

class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")
    
    def get_train_examples_text(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "dev.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")
    
    def get_dev_examples_text(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "dev.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")
    
    def get_test_examples_text(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")
    
    def get_labels(self):
        return os.getenv("LABELS", "").split(",")
        # # For vlsp2018
        # return ["I-ORGANIZATION","B-ORGANIZATION","I-LOCATION","B-MISCELLANEOUS","I-PERSON","O","B-PERSON","I-MISCELLANEOUS","B-LOCATION","E","X","<s>","</s>"]

        # # For vlsp2021
        # return ["I-PRODUCT-AWARD","B-MISCELLANEOUS","B-QUANTITY-NUM","B-ORGANIZATION-SPORTS","B-DATETIME","I-ADDRESS","I-PERSON","I-EVENT-SPORT","B-ADDRESS","B-EVENT-NATURAL","I-LOCATION-GPE","B-EVENT-GAMESHOW","B-DATETIME-TIMERANGE","I-QUANTITY-NUM","I-QUANTITY-AGE","B-EVENT-CUL","I-QUANTITY-TEM","I-PRODUCT-LEGAL","I-LOCATION-STRUC","I-ORGANIZATION","B-PHONENUMBER","B-IP","O","B-QUANTITY-AGE","I-DATETIME-TIME","I-DATETIME","B-ORGANIZATION-MED","B-DATETIME-SET","I-EVENT-CUL","B-QUANTITY-DIM","I-QUANTITY-DIM","B-EVENT","B-DATETIME-DATERANGE","I-EVENT-GAMESHOW","B-PRODUCT-AWARD","B-LOCATION-STRUC","B-LOCATION","B-PRODUCT","I-MISCELLANEOUS","B-SKILL","I-QUANTITY-ORD","I-ORGANIZATION-STOCK","I-LOCATION-GEO","B-PERSON","B-PRODUCT-COM","B-PRODUCT-LEGAL","I-LOCATION","B-QUANTITY-TEM","I-PRODUCT","B-QUANTITY-CUR","I-QUANTITY-CUR","B-LOCATION-GPE","I-PHONENUMBER","I-ORGANIZATION-MED","I-EVENT-NATURAL","I-EMAIL","B-ORGANIZATION","B-URL","I-DATETIME-TIMERANGE","I-QUANTITY","I-IP","B-EVENT-SPORT","B-PERSONTYPE","B-QUANTITY-PER","I-QUANTITY-PER","I-PRODUCT-COM","I-DATETIME-DURATION","B-LOCATION-GPE-GEO","B-QUANTITY-ORD","I-EVENT","B-DATETIME-TIME","B-QUANTITY","I-DATETIME-SET","I-LOCATION-GPE-GEO","B-ORGANIZATION-STOCK","I-ORGANIZATION-SPORTS","I-SKILL","I-URL","B-DATETIME-DURATION","I-DATETIME-DATE","I-PERSONTYPE","B-DATETIME-DATE","I-DATETIME-DATERANGE","B-LOCATION-GEO","B-EMAIL","E","X","<s>","</s>"]
    
        # For vlsp2016
        return ["B-ORG","B-MISC","I-PER","I-ORG","B-LOC","I-MISC","I-LOC","O","B-PER","E","X","<s>","</s>"]

    def get_auxlabels(self):
        return ["O", "B", "I","E", "X", "<s>", "</s>"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['<s>']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['</s>']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        if imgs == []:
            for i, (sentence, label) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(sentence)
                text_b = None
                label = label
                auxlabel = auxlabels[i]
                examples.append(
                    SBInputExampleText(guid=guid, text_a=text_a, text_b=text_b, label=label, auxlabel=auxlabel))
        else:
            for i, (sentence, label) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(sentence)
                text_b = None
                label = label
                img_id = imgs[i]
                auxlabel = auxlabels[i]
                examples.append(
                    SBInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
        return examples


def convert_mm_examples_to_features_text(examples, label_list, auxlabel_list,
 max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens2 = []
        segment_ids2 = []
        label_ids2 = []

        auxlabel_ids = []
        ntokens.append("<s>")
        segment_ids.append(0)
        label_ids.append(label_map["<s>"])
        auxlabel_ids.append(auxlabel_map["<s>"])

        segment = 0
        flag = True
        for i, token in enumerate(tokens):
            if token != "</s>" and flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                ntokens2.append(token)
                segment_ids2.append(0)
                label_ids2.append(label_map[labels[i]])
            elif token != "</s>" and not flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
            elif token == "</s>":
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                segment+=1
                flag = False

        ntokens.append("</s>")
        segment_ids.append(segment)
        label_ids.append(label_map["</s>"])
        auxlabel_ids.append(auxlabel_map["</s>"])

        ntokens2.append("</s>")
        segment_ids2.append(0)
        label_ids2.append(label_map["</s>"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        input_ids2 = tokenizer.convert_tokens_to_ids(ntokens2)
        input_mask2 = [1] * len(input_ids2)

        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(segment)
            label_ids.append(0)
            auxlabel_ids.append(0)

        while len(input_ids2) < max_seq_length:
            input_ids2.append(1)
            input_mask2.append(0)
            segment_ids2.append(0)
            label_ids2.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        assert len(input_ids2) == max_seq_length
        assert len(input_mask2) == max_seq_length
        assert len(segment_ids2) == max_seq_length
        assert len(label_ids2) == max_seq_length

        assert len(auxlabel_ids) == max_seq_length


        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("input_mask2: %s" % " ".join([str(x) for x in input_mask2]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("segment_ids2: %s" % " ".join([str(x) for x in segment_ids2]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_ids2: %s" % " ".join([str(x) for x in label_ids2]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            SBInputFeaturesText(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                label_id=label_ids, label_id2=label_ids2, auxlabel_id=auxlabel_ids))

    print('the number of problematic samples: ' + str(count))
    return features

def convert_mm_examples_to_features(examples, label_list, auxlabel_list,
 max_seq_length, tokenizer, path_img, image_feat_model):

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir='cache')# Load the image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path_fail = os.path.join(path_img, 'background.jpg')
    image_path_fail = Image.open(image_path_fail)
    inputs_image_path_fail = processor(images=image_path_fail, return_tensors="pt", padding=True)
    inputs_image_path_fail = {key: value.to(device) for key, value in inputs_image_path_fail.items()}
    image_path_fail_features = image_feat_model.get_image_features(**inputs_image_path_fail)

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens2 = []
        segment_ids2 = []
        label_ids2 = []

        auxlabel_ids = []
        ntokens.append("<s>")
        segment_ids.append(0)
        label_ids.append(label_map["<s>"])
        auxlabel_ids.append(auxlabel_map["<s>"])

        segment = 0
        flag = True
        for i, token in enumerate(tokens):
            if token != "</s>" and flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                ntokens2.append(token)
                segment_ids2.append(0)
                label_ids2.append(label_map[labels[i]])
            elif token != "</s>" and not flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
            elif token == "</s>":
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                segment+=1
                flag = False

        ntokens.append("</s>")
        segment_ids.append(segment)
        label_ids.append(label_map["</s>"])
        auxlabel_ids.append(auxlabel_map["</s>"])

        ntokens2.append("</s>")
        segment_ids2.append(0)
        label_ids2.append(label_map["</s>"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        input_ids2 = tokenizer.convert_tokens_to_ids(ntokens2)
        input_mask2 = [1] * len(input_ids2)

        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(segment)
            label_ids.append(0)
            auxlabel_ids.append(0)

        while len(input_ids2) < max_seq_length:
            input_ids2.append(1)
            input_mask2.append(0)
            segment_ids2.append(0)
            label_ids2.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        assert len(input_ids2) == max_seq_length
        assert len(input_mask2) == max_seq_length
        assert len(segment_ids2) == max_seq_length
        assert len(label_ids2) == max_seq_length

        assert len(auxlabel_ids) == max_seq_length
        
        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)
        if not os.path.exists(image_path):
            if 'NaN' not in image_path:
                pass
                # print(image_path)
        try:
            image_path = Image.open(image_path)
            input_image = processor(images=image_path, return_tensors="pt", padding=True)
            input_image = {key: value.to(device) for key, value in input_image.items()}
            img_feat = image_feat_model.get_image_features(**input_image)
        except:
            count += 1
            img_feat = image_path_fail_features

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("input_mask2: %s" % " ".join([str(x) for x in input_mask2]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("segment_ids2: %s" % " ".join([str(x) for x in segment_ids2]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_ids2: %s" % " ".join([str(x) for x in label_ids2]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            SBInputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2, img_feat=img_feat, label_id=label_ids, label_id2=label_ids2, auxlabel_id=auxlabel_ids))

    del inputs_image_path_fail
    del image_path_fail_features
    gc.collect()  # Trigger garbage collection to release unused memory
    torch.cuda.empty_cache()  # Clear cached GPU memory

    print('the number of problematic samples: ' + str(count))
    return features


if __name__ == "__main__":
    processor = MNERProcessor()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1


    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    data_dir = r'sample_data\BC5CDR-disease-IOB'
    train_examples = processor.get_train_examples(data_dir)