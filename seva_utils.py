import nltk, re
import nltk
from nltk.corpus import stopwords

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
        d[splits[0]] = " ".join(splits[1:])
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
    d = fix_def(accr_location, definition_location)
    res = {}
    for key in d.keys():
        splits = key.split("|")
        for s in splits:
            s = s.strip()
            if s == "":
                continue
            if not s.isupper():
                if res.get(len(s.split()), None) is None:
                    res[len(s.split())] = set()
                res[len(s.split())].add(s)
    return res


######## USAGE EXAMPLE ########
#accr_location = "se_data/acronyms.txt"
#definition_location = "se_data/definitions.txt"
#d = fix_def(accr_location, definition_location)
#res = categorize_def_into_numwords(accr_location, definition_location)

