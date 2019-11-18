import pandas as pd
from sklearn.utils import shuffle
import sys, re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import punkt
#stop_words = stopwords.words('english')
import numpy as np
import re, random
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.chunk import RegexpParser
import nltk, scipy, emoji
from nltk.corpus import wordnet
import csv, sys, random, math, re, itertools
from sklearn.utils import shuffle
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import enchant
d = enchant.Dict("en_US")

snowball_stemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer()

def remove_adjacent_duplicates(word_list):
    curr = None
    new_word_list = []
    for i in range(len(word_list)):
        if curr is None:
            curr = word_list[i]
            new_word_list.append(curr)
            continue
        if word_list[i] != curr:
            curr = word_list[i]
            new_word_list.append(curr)
    return new_word_list

def remove_adjacent_duplicates_fromline(line):
    word_list = line.split()
    #tknzr = TweetTokenizer()
    #word_list = tknzr.tokenize(line)
    #new_word_list = [word for word in word_list if len(word) > 2]
    return ' '.join(remove_adjacent_duplicates(word_list))

def preprocess_0(sentence):

    if type(sentence) != str:
        return ""
    
    sentence = (sentence.encode('ascii', 'ignore')).decode("utf-8")
    
    # URLs
    sentence = re.sub(r'http\S+', ' <URL> ', sentence)
    
    # emoji
    for c in sentence:
        if c in emoji.UNICODE_EMOJI:
            sentence = re.sub(c, emoji.demojize(c), sentence)
    
    sentence = re.sub("([!]){1,}", " ! ", sentence)
    sentence = re.sub("([.]){1,}", " . ", sentence)
    sentence = re.sub("([?]){1,}", " ? ", sentence)
    sentence = re.sub("([;]){1,}", " ; ", sentence)
    sentence = re.sub("([:]){2,}", " : ", sentence)
    
    # separate characters
    sentence = re.sub("[;|.|:|?|!|/|&|*|+|-|%|>|<]", " ", sentence)
    
    # convert words such as "goood" to "good"
    sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))
    
    sentence = re.sub('''[^ a-zA-Z0-9.,!'"?:;<>&\()\-\n]''', ' ', sentence)
    #sentence = re.sub('\s+', ' ', sentence)
    
    return remove_adjacent_duplicates_fromline(sentence)

def preprocess_pos(sentence):
    
    sentence = preprocess_1(sentence)

    pos_tags = nltk.pos_tag(tknzr.tokenize(sentence))

    words = []
    for word,tag in pos_tags:
        words.append(irr_word_2(word, tag))

    words = [word for word in words if d.check(word.lower()) or len(word) <=1 or not word[0].isalpha()]
    
    new_sent = " ".join(words)

    new_sent = preprocess_2(new_sent)

    return remove_adjacent_duplicates_fromline(new_sent.strip())


def get_abbr_parts(abbr):
    chars = []
    start = 0
    for index in range(1,len(abbr)):
        if abbr[index].isupper():
            chars.append(abbr[start:index])
            start = index
    chars.append(abbr[start:])
    return chars
            
def get_full_form(subtext, abbr_parts, flag=True):
    ff = []
    words = list(reversed(subtext.split()))
    abbr_parts = list(reversed(abbr_parts))
    abbr_parts = [abr[0] for abr in abbr_parts]
    words_index = 0
    index = 0
    while index < len(abbr_parts):
        if words_index > len(words)-1:
            ff = []
            break
        if not abbr_parts[index].isalpha():
            index += 1
            continue
        #print(index)
        #print(words[words_index])
        #print(abbr_parts[index])
        if words_index == 0:
            if abbr_parts[index].lower() != words[words_index][0].lower():
                ff = []
                break
        if abbr_parts[index].lower() != words[words_index][0].lower() and len(words[words_index]) > 3:
            ff = []
            break
        if abbr_parts[index].lower() != words[words_index][0].lower() and len(words[words_index]) <= 3:
            ff.append(words[words_index])
            index -= 1
            words_index += 1
        elif abbr_parts[index].lower() == words[words_index][0].lower() and len(words[words_index]) >= 3:
            ff.append(words[words_index])
            words_index += 1
        elif abbr_parts[index].lower() == words[words_index][0].lower() and len(words[words_index]) < 3:
            if flag:
                ff.append(words[words_index])
                words_index += 1
            else:
                ff.append(words[words_index])
                index -= 1
                words_index += 1
        index += 1
    
    ff = list(reversed(ff))
    return " ".join(ff).strip()
    
    
def get_abrreviations(text):
    abbr_dict = {}
    abbrs = re.findall(r"\([ ]*[A-Z][&A-Za-z]*[ ]*\)",text)
    print(abbrs)
    
    for abbr in abbrs:
        a = abbr[1:len(abbr)-1].strip()
        abbr_parts = get_abbr_parts(a)
        subtext = text[:text.index(abbr)].strip()
        full_abbr = get_full_form(subtext, abbr_parts)
        if full_abbr != "":
            abbr_dict[a] = full_abbr
        full_abbr = get_full_form(subtext, abbr_parts, False)
        if full_abbr != "":
            abbr_dict[a] = full_abbr
    return abbr_dict
        

