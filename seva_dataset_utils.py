import nltk, re
from nltk.corpus import stopwords
import enchant
from nltk import ngrams
d = enchant.Dict("en_US")

# create a dictionary
# input: accronym file
# output: dictionary
# key: acronym
# value: full form
def create_acronym_dict(fname):
    d = {}
    f = open(fname)
    lines = f.readlines()
    f.close()
    for line in lines:
        splits = line.strip().split()
        if len(splits) < 2:
            continue
        d[splits[0].strip()] = " ".join(splits[1:]).strip()
    return d

# create a dictionary
# input: accronym file
# output: dictionary
# key: full form
# value: accronym
def create_acronym_dict_inverse(fname):
    d = {}
    f = open(fname)
    lines = f.readlines()
    f.close()
    for line in lines:
        splits = line.strip().split()
        if len(splits) < 2:
            continue
        d[" ".join(splits[1:]).strip().lower()] = splits[0].strip()
    return d

# create a dictionary
# input: definition file
# output: dictionary
# key: definition
# value: description
def create_definition_dict(fname):
    d = {}
    f = open(fname)
    lines = f.readlines()
    f.close()
    for line in lines:
        splits = line.strip().split(":")
        d[splits[0]] = " ".join(splits[1:])
    return d

# dealing with definitions with '/' and accronyms
# Automated/Automation
# Configuration Items (CI)
# input: phrase/definition
# output: list of definitions extracted from the input
def get_def_combos(term, accro_dict):
    in_brackets  = re.findall(r"\([^\(\)]*\)",term)
    term = re.sub(r"\([^\(\)]*\)", " ", term)
    term = re.sub('\s+', ' ', term)
    in_brackets = [b[1:len(b)-1] for b in in_brackets]
    in_brackets = [b.strip() for b in in_brackets if accro_dict.get(b, None) is not None]
    terms_plus = term.split("/")
    terms_plus = [t.lower().strip() for t in terms_plus]
    #in_brackets = [b.lower().strip() for b in in_brackets if accro_dict.get(b).lower() in terms_plus]
    combo = terms_plus + in_brackets
    return combo
    
# Fixing the first sentence of the explanation to create a coherent sentence
# input: accronym file, definition file
# output: dictionary
# key: definition
# value: full form
def fix_def(accr_fname, fname):
    accro_dict = create_acronym_dict(accr_fname)
    f = open(fname)
    lines = f.readlines()
    f.close()
    
    definition_dict = {}
    for line in lines:
        sent = nltk.sent_tokenize(line.strip())
        if len(sent) < 1:
            continue
        splits = sent[0].strip().split(":")
        definition = splits[0]
        explanation = splits[1:]
        explanation = explanation[0]
        combos = get_def_combos(definition, accro_dict)
        new_def = " | ".join(combos)
        Flag = False
        for c in combos:
            if c.lower() in explanation.lower():
                Flag = True
        if Flag:
            if len(sent) > 1:
                explanation += " ".join(sent[1:])
            definition_dict[new_def.strip()] = explanation.strip()
        else:
            if explanation.split()[0].lower() in stopwords.words('english'):
                explanation = explanation.split()
                explanation[0] = explanation[0].lower()
            else: 
                explanation = explanation.split()
                explanation[0] = explanation[0].lower()
            exp = definition + " is " + " ".join(explanation).strip()
            if len(sent) > 1:
                exp += " ".join(sent[1:])
            definition_dict[new_def.strip()] = exp
    return definition_dict

# put definitions in to buckets corresponding to number of words in them
# input: accronym filename, definition filename
# output: dictionary
# key: integer 'n' - number of words
# value: definitions that has 'n' number of words
def categorize_def_into_numwords(accr_location, definition_location):
    accro_dict = create_acronym_dict(accr_location)
    d = fix_def(accr_location, definition_location)
    res = {}
    for key in d.keys():
        splits = key.split("|")
        for s in splits:
            s = s.strip()
            if s == "":
                continue
            if not s.isupper() and s not in accro_dict.keys():
                if res.get(len(s.split()), None) is None:
                    res[len(s.split())] = set()
                res[len(s.split())].add(s)
    return res

# write all keywords from buckets to a file
# input: accronym filename, definition filename, output filename
# output: a new file
def create_file_from_buckets_4_human_labelling(accr_location, definition_location, fname):
    res = categorize_def_into_numwords(accr_location, definition_location)
    f = open(fname, 'w+')
    keys = sorted(res.keys())
    for key in keys:
        f.write(str(key)+'\n')
        for item in res[key]:
            f.write(item+", \n")
    f.close()

# read annotated keywords as dictionary
# input: filename
# ouput: dictionary
# key: tag
# value: terms
def read_annotated_keywords(fname):
    tag_dict = {}
    f = open(fname)
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line.split()) > 1:
            tag = line.split(",")[-1].strip()
            term = " ".join(line.split(",")[:-1]).strip()
            if tag_dict.get(tag, None) is None:
                tag_dict[tag] = set()
            tag_dict[tag].add(term.strip())

    '''
    accro_dict = create_acronym_dict_inverse(accr_location)
    for key,val in tag_dict.items():
        new_item_set = set()
        for item in val:
            splits = item.split()
            for index in range(len(splits)):
                if accro_dict.get(splits[index].strip().lower(), None) is not None:
                    splits[index] = accro_dict.get(splits[index].strip().lower())
            new_item = " ".join(splits)
            new_item_set.add(new_item)
        tag_dict[key] = tag_dict[key].union(new_item_set)
    '''
    return tag_dict

# read annotated keywords as dictionary
# input: filename
# ouput: dictionary
# key: term
# value: tag
def read_annotated_keywords_inverse(fname):
    tag_dict = {}
    f = open(fname)
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line.split()) > 1:
            tag = line.split(",")[-1].strip()
            term = " ".join(line.split(",")[:-1]).strip()
            tag_dict[term.strip().lower()] = tag
    return tag_dict

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def update_vocab(location_of_vocab, *input_fname):
    text = ""
    for fname in input_fname:
        f_read = open(fname)
        text += " " + f_read.read()
        f_read.close()
    new_words = nltk.word_tokenize(text)
    new_words = [w.strip().lower() for w in new_words]
    new_words = set(new_words)
    n = len(new_words)

    f = open(location_of_vocab)
    text = f.readlines()
    f.close()
    old_words = [w.strip() for w in text]

    o_len = len(old_words)
    old_words = set(old_words)

    words = old_words.union(new_words)

    num_words2remove = len(words) - o_len

    while num_words2remove > 0 :
        num_words = len(words)
        count = 1
        for item in words:
            count += 1
            if not is_ascii(item):
                words.remove(item)
                num_words2remove -= 1
                break
        if count >= num_words:
            break

    unused = 993

    while num_words2remove > 0 and unused >=0 :
        for item in words:
            if 'unused' in item:
                words.remove(item)
                num_words2remove -= 1
                unused -= 1
                break

    assert(o_len == len(words))

    f = open(location_of_vocab, 'w+')
    for word in words:
        f.write(word)
        f.write('\n')
    f.close()

def sentence2tag_keyword(sentence, keyword_fname):
    phrase_labels = []
    d = read_annotated_keywords_inverse(keyword_fname)
    for i in reversed(range(1, 6)):
        words = nltk.word_tokenize(sentence)
        igrams = ngrams(words, i)
        for g in igrams:
            phrase = " ".join(g).strip()
            if d.get(phrase.lower(), None) != None:
                phrase_labels.append((phrase,d[phrase.lower()]))
                sentence = sentence.replace(phrase, "").strip()
    return phrase_labels


def keywords2sentposlabel(keyword_fname):
    d = read_annotated_keywords_inverse(keyword_fname)
    sentences = []
    poses = []
    labels = []
    for key,val in d.items():
        words = nltk.word_tokenize(key)
        #words.append(".")
        pos = nltk.pos_tag(words)
        pos = [p[1] for p in pos]
        lab = []
        for index in range(len(words)):
            if index == 0:
                lab.append('B-'+val)
            else:
                lab.append('I-'+val)
        sentences.append(words)
        poses.append(pos)
        labels.append(lab)
    return sentences, poses, labels
    


######## USAGE EXAMPLE ########
# accr_location = "se_data/acronyms.txt"
# definition_location = "se_data/definitions.txt"
# d = fix_def(accr_location, definition_location)
# res = categorize_def_into_numwords(accr_location, definition_location)
# create_file_from_buckets_4_human_labelling(accr_location, definition_location, "se_data/keywords2annotate.txt")
# d = read_annotated_keywords( accr_location, "se_data/keywords2annotate.txt") #text file should be annotated
# d = read_annotated_keywords_inverse("se_data/keywords2annotate.txt")
