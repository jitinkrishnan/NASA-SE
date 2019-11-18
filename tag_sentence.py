
# reference: https://medium.com/@yingbiao/ner-with-bert-in-action-936ff275bc73

import pandas as pd
import math, nltk, re
import numpy as np
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import BertForTokenClassification, AdamW
from seva_dataset_utils import *
from seva_preprocess import *
from extract_verb_phrase import *

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence#").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def tag_sent(text):
	# initialize variables
	num_tags = 24 # depends on the labelling scheme
	max_len  = 45
	vocabulary = "bert_models/vocab.txt"
	bert_out_address = 'bert/model'

	tokenizer=BertTokenizer(vocab_file=vocabulary,do_lower_case=False)

	model = BertForTokenClassification.from_pretrained(bert_out_address,num_labels=num_tags)

	f = open('se_data/tags.txt')
	lines = f.readlines()
	f.close()

	tag2idx = {}
	for line in lines:
		key = line.split()[0]
		val = line.split()[1]
		tag2idx[key.strip()] = int(val.strip())

	tag2name={tag2idx[key] : key for key in tag2idx.keys()}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()

	if torch.cuda.is_available():
		model.cuda();
		if n_gpu >1:
			model = torch.nn.DataParallel(model)

	model.eval();

	tokenized_texts = []
	word_piece_labels = []
	i_inc = 0

	temp_token = []
	
	# Add [CLS] at the front 
	temp_token.append('[CLS]')
	
	for word in nltk.word_tokenize(text):
		token_list = tokenizer.tokenize(word)
		for m,token in enumerate(token_list):
			temp_token.append(token)


	# Add [SEP] at the end
	temp_token.append('[SEP]')
	
	tokenized_texts.append(temp_token)
	
	#if 5 > i_inc:
		#print("No.%d,len:%d"%(i_inc,len(temp_token)))
		#print("texts:%s"%(" ".join(temp_token)))
	#i_inc +=1

	input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")

	attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
	#attention_masks[0];

	segment_ids = [[0] * len(input_id) for input_id in input_ids]
	#segment_ids[0];

	tr_inputs = torch.tensor(input_ids).to(device)
	tr_masks = torch.tensor(attention_masks).to(device)
	tr_segs = torch.tensor(segment_ids).to(device)

	outputs = model(tr_inputs, token_type_ids=None, attention_mask=tr_masks,)

	#tr_masks = tr_masks.to('cpu').numpy()

	logits = outputs[0] 

	# Get NER predict result
	logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
	logits = logits.detach().cpu().numpy()

	#print(logits)
	#print(len(logits[0]))
	tags_t = [tag2name[t] for t in logits[0]]

	#print(nltk.word_tokenize(text))
	c = len(tokenized_texts[0])
	#print(tags_t[:c])
	return tokenized_texts[0][1:len(temp_token)-1], tags_t[:c][1:len(tags_t[:c])-1]

# should follow the same ner tagging format
def ner_excel2lists(fname):
	df_data = pd.read_csv(fname,sep="\t",encoding="latin1").fillna(method='ffill')
	df_data['Sentence#'].nunique(), df_data.Word.nunique(), df_data.POS.nunique(), df_data.Tag.nunique()

	getter = SentenceGetter(df_data)
	sentences = [[s[0] for s in sent] for sent in getter.sentences]
	poses = [[s[1] for s in sent] for sent in getter.sentences]
	labels = [[s[2] for s in sent] for sent in getter.sentences]

	return sentences, poses, labels

def identify_entity_tag(sentence_lst, label_lst):
    sent_labl = []
    tag = 'O'
    words_sofar = []
    for index in range(len(sentence_lst)):
        if label_lst[index][0] == 'B':
            if tag != 'O':
                sent_labl.append((" ".join(words_sofar),tag))
            words_sofar = []
            tag = label_lst[index][2:]
            words_sofar.append(sentence_lst[index])
        elif label_lst[index][0] == 'I':
            words_sofar.append(sentence_lst[index])
        else:
            if tag != 'O':
                sent_labl.append((" ".join(words_sofar),tag))
            words_sofar = []
            tag = 'O'
    return " ".join(sentence_lst), sent_labl

def tag_check(tag):
    selected_tag = ['opcon', 'seterm', 'syscon', 'art']
    for s in selected_tag:
        if s in tag:
            return True
    return False

def identify_entity_tag_extra(sentence_lst, pos_lst, label_lst):
    sent_labl = []
    tag = 'O'
    words_sofar = []
    for index in range(len(sentence_lst)):
        if label_lst[index][0] == 'B':
            if tag != 'O':
                sent_labl.append((" ".join(words_sofar),tag))
            words_sofar = []
            tag = label_lst[index][2:]
            words_sofar.append(sentence_lst[index])
        elif label_lst[index][0] == 'I':
            words_sofar.append(sentence_lst[index])
        else:
            if tag != 'O':
                if index > 0:
                    if re.match("N{1}[.]*",pos_lst[index]) and tag_check(tag):
                        words_sofar.append(sentence_lst[index])
                    else:
                        sent_labl.append((" ".join(words_sofar),tag))
                        words_sofar = []
                        tag = 'O'
                else:
                    sent_labl.append((" ".join(words_sofar),tag))
                    words_sofar = []
                    tag = 'O'
            #words_sofar = []
            #tag = 'O'
    return sent_labl

def sentence2tags(text):
	text = preprocess_0(text)
	word_list, label_list = tag_sent(text)
	pos_tags = nltk.pos_tag(word_list)
	poses = [p[1] for p in pos_tags]
	return identify_entity_tag_extra(word_list, poses, label_list)

def checkphrase_inlist(phrase, mylist):
	words = nltk.word_tokenize(phrase)
	for word in words:
		if word not in mylist and word.lower() not in mylist:
			return False
	return True

def get_indices(words, phrase):
	winphrase = nltk.word_tokenize(phrase)
	indices = []
	for index in range(len(words)):
		if words[index] == winphrase[0]:
			j = index +1
			for i in range(1, len(winphrase)):
				if words[j] != winphrase[i]:
					break
				j += 1
			if j-index == len(winphrase):
				indices.append((index, index+len(winphrase)-1))
	return indices



def sentence2tags_all(text):
	text = preprocess_0(text)
	word_list, label_list = tag_sent(text)
	pos_tags = nltk.pos_tag(word_list)
	poses = [p[1] for p in pos_tags]
	keyword_fname = "se_data/keywords2annotate.txt"

	words =  nltk.word_tokenize(text)

	term_label = sentence2tag_keyword(text, keyword_fname)
	term_label = [t for t in term_label if checkphrase_inlist(t[0], words)]

	sentlabel = identify_entity_tag_extra(word_list, poses, label_list)
	sentlabel  = [t for t in sentlabel  if checkphrase_inlist(t[0], words)]

	sentlabel_phrases = [t[0] for t in sentlabel]
	term_label = [t for t in term_label if t[0] not in sentlabel_phrases and t[0].lower() not in sentlabel_phrases]

	sentlabel = term_label + sentlabel

	return sentlabel

def getkey(item): 
    return item[0][0]

def verb_phrase_relations(text):
	sentence_labels = sentence2tags_all(text)
	#print("SL: ",  sentence_labels)
	text = preprocess_0(text)
	words = nltk.word_tokenize(text)
	pos_tags = nltk.pos_tag(words)

	indices_tags = []
	for sent in sentence_labels:
		indices = get_indices(words, sent[0])
		for index in indices:
			indices_tags.append((index, sent[1]))
	indices_tags = sorted(indices_tags, key = getkey) 

	if len(indices_tags) < 2:
		return []

	relations = []
	for index in range(1, len(indices_tags)):
		t1 = indices_tags[index-1][0]
		t2 = indices_tags[index][0]
		s_tag = " ["+indices_tags[index-1][1]+"]"
		t_tag = " ["+indices_tags[index][1]+"]"
		if t1[1]+1 >= t2[0]:
			continue
		source = " ".join(words[t1[0]:t1[1]+1])
		target = " ".join(words[t2[0]:t2[1]+1])
		_ ,rels = extract_vp(pos_tags[t1[1]+1:t2[0]-1])
		for rel in rels:
			relations.append((source+s_tag, rel, target+t_tag))
	return relations


