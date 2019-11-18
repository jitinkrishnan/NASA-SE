from seva_dataset_utils import *
from nltk import ngrams
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()


def find_super(phrase):
    words = nltk.word_tokenize(phrase)
    if len(words) < 2:
        return None
    pos_tags = nltk.pos_tag(words)
    index = len(words)-1
    tags = [t[1] for t in pos_tags]
    if 'IN' in tags:
        index = 0
    return lemmatizer.lemmatize(words[index])

def phrase_ngrams(phrase):
    words = nltk.word_tokenize(phrase)
    grams = []
    for i in range(1, len(words)+1):
        igrams = ngrams(words, i)
        for g in igrams:
            grams.append(" ".join(g).strip())
    return grams

def create_relations_from_accronym_def(accr_location, definition_location):
	res = categorize_def_into_numwords(accr_location, definition_location)

	entities = set()

	source = []
	target = []
	relations = []

	for i in sorted(res.keys()):
		new_entities = set()
		for item in res[i]:
			if i == 1:
				item = lemmatizer.lemmatize(item)
			new_entities.add(item)
			s = find_super(item)
			if s is not None:
				if s not in entities:
					entities.add(s)       
				for phrase in phrase_ngrams(item):
					if phrase in entities:
						source.append(item)
						target.append(phrase)
						relations.append("part-of")
		entities = entities.union(new_entities)

	return source, target, relations, entities