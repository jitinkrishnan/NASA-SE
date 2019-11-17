#Program to extract triples from simple sentences

import nltk, re, sys
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import RegexpParser
from nltk.tag.stanford import StanfordPOSTagger

#pos_tagger
path_to_jar = 'stanford_jars/stanford-postagger-3.9.2.jar'
path_to_models_jar = 'stanford_jars/models/english-bidirectional-distsim.tagger'

stanford_pos_tagger = StanfordPOSTagger(path_to_models_jar, path_to_jar)

#dependency parser
path_to_jar = 'stanford_jars/stanford-parser.jar'
path_to_models_jar = 'stanford_jars/stanford-parser-3.9.2-models.jar'

stanford_dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def checkTripleForCOP(triple):
    (a,b,c) = triple
    r = re.compile(r'VB.*')
    (d,e) = c
    if b=='cop' and d == 'is' and r.match(e):
        return a[0]
    return None

def checkNounNSUBJ(triple):
    (a,b,c) = triple
    (w1,t1) = a
    (w2,t2) = c
    r = re.compile(r'NN.*')
    rc = re.compile(r'CD.*')
    if b == 'nsubj' and (r.match(t1) or rc.match(t1)) and (r.match(t2) or rc.match(t2)):
        return True
    return False

#Triple extraction method for elementary S-V-O relation
def SVO(dep_triples):
    triple_array = []
    r = re.compile(r'VB.*')
    for t in dep_triples:
        (x,y,z) = t
        (p,pt) = x
        if(r.match(pt) and y == 'nsubj'):
            (s1,s2) = z
            subject = s1
            rel = p
            for t2 in dep_triples:
                (x2,y2,z2) = t2
                (p2,pt2) = x2
                if(x == x2 and y2 == 'dobj'):
                    (o1,o2) = z2
                    tri = (subject, rel, o1)
                    if tri:
                        triple_array.append(tri)
    return triple_array


#Triple extraction method for IS-A relation
def IS_A(dep_triples):
    triple_array = []
    r = re.compile(r'VB.*')
    cop = None;
    nsubj1 = None
    nsubj2 = None
    for t in dep_triples:
        if cop is None:
            cop = checkTripleForCOP(t)
        (x,y,z) = t
        if y == 'appos':
            triple_array.append((x[0],'is-a',z[0]))
        if checkNounNSUBJ(t):
            nsubj1 = z[0]
            nsubj2 = x[0]
    if nsubj1 != None and nsubj2 != None and cop != None:
        triple_array.append((nsubj1,'is',cop))
    return triple_array
    
#Insert word
def word_insert(phrase, word1, word2):
    index = phrase.find(word1)
    newPhrase = phrase[:index]+word2+' '+phrase[index:]
    return newPhrase
    

#Triple extraction method for compound words
def compound(dep_triples, SVO):
    (s,v,o) = SVO
    s1 = s
    o1 = o
    for t in dep_triples:
        (x,y,z) = t
        if y == 'compound' or y == 'nummod' or y == 'npmod' or (y == 'nmod' and z[1] is 'CD'):
            if x[0] in s:
                s1 = word_insert(s1,x[0],z[0])
                #s1 = z[0]+' '+s
            if x[0] in o:
                o1 = word_insert(o1,x[0],z[0])
                #o1 = z[0]+' '+o
    return (s1,v,o1)
    
#Method to find index into tagger to find a word
def tag_index(tagged_sentence, myWord):
    for token in nltk.word_tokenize(myWord):
        for i in range(len(tagged_sentence)):
            word_tag = tagged_sentence[i]
            if token == word_tag[0]:
                return i
    return -1

#Method to find triple from chunks
def chunk_triples(tagged_sentence, index):
    word = tagged_sentence[index][0]
    triple_array = []
    adj = re.compile(r'JJ.*')
    cd = re.compile(r'CD.*')
    for i in range(index):
        word_tag = tagged_sentence[i]
        if adj.match(word_tag[1]):
            triple_array.append((word, 'has-property', word_tag[0]))
        #if cd.match(word_tag[1]):
            #triple_array.append((word, 'has-value', word_tag[0]))
    return triple_array    

##Triple extraction method for additional relations
def additionalExtractions(dep_triples, tagged_sentence, svo_triples):
    if not svo_triples:
        return None
    grammar = "SmallNP: {(<CD.*>|<JJ.*>)<NN.*>+}"
    cp = RegexpParser(grammar)
    chunk = cp.parse(tagged_sentence)
    triple_array = []
    for subtree in chunk.subtrees():
        if subtree.label() == 'SmallNP':
            for triple in svo_triples:
                pos = subtree.leaves()
                loc1 = tag_index(pos,triple[0])
                if loc1 != -1:
                    triple_array.extend(chunk_triples(pos,loc1))
                loc2 = tag_index(pos,triple[2])
                if loc2 != -1:
                    triple_array.extend(chunk_triples(pos,loc2))
    return triple_array

#passive-verb extraction
def PASSIVE_VOICE(triple1, triple2):
    if triple1[1] == 'nsubjpass' and triple2[1] == 'auxpass' and triple1[0] == triple2[0]:
        return (triple1[2][0],triple2[2][0],triple1[0][0])
    return None

#AMOD ADJECTIVE-OR-NOUN relation
def AMOD_ADJECTIVE(triple):
    if triple[1] == 'amod':
        return (triple[0][0], 'has-property', triple[2][0])
    return None
    
#NMOD-CASE relations-1,2
def NMOD_CASE_1(triple1, triple2):
    r = re.compile(r'VB.*')
    if triple1[1] == 'nmod' and triple2[1] == 'case' and triple1[2] == triple2[0] and r.match(triple1[0][1]):
        return (triple1[0][0], triple2[2][0],triple1[2][0])
    return None

def NMOD_CASE_2(triple1, triple2):
    if (triple1[1] == 'nmod' and triple2[1] == 'case' and triple2[2][0] == 'of') or triple1[1] == 'nmod:poss':
       r = re.compile(r'NN.*') #only for Nouns
       if r.match(triple1[2][1]):
           return (triple1[2][0],'has-property', triple1[0][0])
       else:
           return ((triple1[0][0], 'has-value', triple1[2][0]))
    return None 

def traverseDepTree(dep_triples):
    important_keywords = ['nsubj', 'dobj', 'nsubjpass']
    added_words = []
    triple_array = []
    for rel in dep_triples:
        if rel[1] in important_keywords:
            added_words.append(rel[2][0])
            added_words.append(rel[0][0])
    
    # PASSIVE RELATIONS - NSUBJPASS, AUXPASS
    # ADJECTIVES - AMOD
    # NMOD-CASE-1
    # NMOD-CASE-2
    for i in range(len(dep_triples)):
        # PASSIVE RELATIONS - NSUBJPASS, AUXPASS
        passive_rel = None
        if i < len(dep_triples)-1:
            passive_rel = PASSIVE_VOICE(dep_triples[i], dep_triples[i+1])
        if passive_rel != None:
            triple_array.append(passive_rel)
            added_words.append(passive_rel[0])
            added_words.append(passive_rel[2])
            i = i+1
            continue

        # ADJECTIVES - AMOD
        if dep_triples[i][0][0] in added_words:
            adj = AMOD_ADJECTIVE(dep_triples[i])
            if adj != None:
                triple_array.append(adj)
                if dep_triples[i][2][0] not in added_words:
                    added_words.append(dep_triples[i][2][0])

        # NMOD-CASE-1
        nmod_case_rel = None
        if i < len(dep_triples)-1 and dep_triples[i][0][0] in added_words:
            nmod_case_rel = NMOD_CASE_1(dep_triples[i], dep_triples[i+1])
        if nmod_case_rel != None:
            triple_array.append(nmod_case_rel)
            added_words.append(nmod_case_rel[0])
            added_words.append(nmod_case_rel[2])
            i = i+1
            continue

        #NMOD-CASE-2
        nmod_case_rel_2 = None
        if i < len(dep_triples)-1 and dep_triples[i][0][0] in added_words:
            nmod_case_rel_2 = NMOD_CASE_2(dep_triples[i], dep_triples[i+1])
        if nmod_case_rel_2 != None:
            triple_array.append(nmod_case_rel_2)
            added_words.append(nmod_case_rel_2[0])
            added_words.append(nmod_case_rel_2[2])
            #i = i+1
        else:
            if 'nmod' in dep_triples[i][1] and dep_triples[i][0][0] in added_words:
                r = re.compile(r'NN.*')
                if r.match(dep_triples[i][2][1]):
                    triple_array.append((dep_triples[i][0][0], 'has-property', dep_triples[i][2][0]))
                else:
                    triple_array.append((dep_triples[i][0][0], 'has-value', dep_triples[i][2][0]))
                if dep_triples[i][2][0] not in added_words:
                    added_words.append(dep_triples[i][2][0])
    
    return triple_array

def relWithNoObjs(dep_triples):
    triple_array = []
    r = re.compile(r'VB.*')
    subj = None
    obj = None
    pred = None
    for t in dep_triples:
        (x,y,z) = t
        (p,pt) = x
        if x[0] == obj and y == 'case' and z[1] == 'IN':
            if pred:
                pred = pred + "-"+z[0]
            else:
                pred = z[0]
        if(r.match(pt) and y == 'nsubj'):
            (s1,s2) = z
            subj = s1
            pred = p
            for t2 in dep_triples:
                (x2,y2,z2) = t2
                (p2,pt2) = x2
                if(x == x2 and y2 == 'nmod'):
                    (o1,o2) = z2
                    obj = o1
    triple = (subj, pred, obj)
    if obj == None: return None
    return triple

#is the triple compound
def checkCOMPOUND(triple):
    if 'compound' in triple[1] or 'nummod' in triple[1] or 'npmod' in triple[1]  and ('nmod' in triple[1] and 'CD' in triple[2][1]):
        return (triple[2][0]+" "+triple[0][0], triple[2][1])
    return None
    
#combine Phrases for compound
def combineWord(phrase1, phrase2):
    myWords1 = phrase1.split()
    if myWords1[len(myWords1)-1] in phrase2:
        new_set = []
        new_set.extend(myWords1[:len(myWords1)-1])
        new_set.extend(phrase2.split())
        return ' '.join(new_set)

#combine triple and word for compound
def COMPOUND(triple, tyuple):
    if checkCOMPOUND(triple) == None:
        myWords1 = triple[0][0].split()
        myWords2 = triple[2][0].split()
        if myWords1[len(myWords1)-1] in tyuple[0]:
            return ((combineWord(triple[0][0],tyuple[0]), tyuple[1]), triple[1], triple[2])
        elif myWords2[len(myWords2)-1] in tyuple[0]:
            return (triple[0], triple[1], ((combineWord(triple[2][0],tyuple[0]), tyuple[1])))
    return None
    
#Compound for subjects and objects
def SVOCompound(dep_triples,word_tag):
    for i in range(len(dep_triples)):
        triple = dep_triples[i]
        new_triple = COMPOUND(triple, word_tag)
        if new_triple != None:
            del dep_triples[i]
            dep_triples.insert(i,new_triple)

#Another function to edit dependency triples - with compound relations
def joinCompoundsInDepTree(dep_triples):
    new_dep_triples = dep_triples.copy()
    size = len(new_dep_triples)
    i = 0
    svo_words = []
    important_keywords = ['nsubj', 'dobj', 'nsubjpass']
    while i < size:
        rel = new_dep_triples[i]
        if rel[1] in important_keywords:
            svo_words.append(rel[2][0])
            svo_words.append(rel[0][0])
        word_tag = checkCOMPOUND(new_dep_triples[i])
        if word_tag != None:
            t = new_dep_triples[i]
            del new_dep_triples[i]
            size = size - 1
            if 'CD' not in word_tag[1] or t[0][0] in svo_words or t[2][0] in svo_words:
                SVOCompound(new_dep_triples,word_tag)
            #flag = False
            #for word in svo_words:
                #if word in t[0][0] or word in t[2][0]:
                    #flag = True
                    #SVOCompound(new_dep_triples,word_tag)
            else:
                if i-1 >=0:
                    triple = new_dep_triples[i-1]
                    new_triple = COMPOUND(triple, word_tag)
                    if new_triple != None:
                        del new_dep_triples[i-1]
                        new_dep_triples.insert(i-1,new_triple)
                if i-2 >=0:
                    triple = new_dep_triples[i-2]
                    new_triple = COMPOUND(triple, word_tag)
                    if new_triple != None:
                        del new_dep_triples[i-2]
                        new_dep_triples.insert(i-2,new_triple)
                i = i-1
        i = i+1
    return new_dep_triples

#Another function to edit dependency triples - with compound relations
def joinCompoundsInDepTree1(dep_triples):
    r = re.compile(r'CD.*')
    new_dep_triples = []
    comp_list = []
    final_dep_triples = []
    for t in dep_triples:
        if t[1] == 'compound' or t[1] == 'nummod' or t[1] == 'npmod' and (t[1] == 'nmod' and t[2][1] == 'CD'):
            comp_list.append(t)
        else: new_dep_triples.append(t)
    for t in new_dep_triples:
        triple = t
        index = new_dep_triples.index(triple)
        del new_dep_triples[index]
        for s in comp_list:
            subj = triple[0]
            rel = triple[1]
            obj = triple[2]
            if s[0][0] in triple[0][0]:
                subj0 = word_insert(triple[0][0],s[0][0],s[2][0])
                subj1 = triple[0][1]
                if r.match(s[2][1]):
                    subj1 = s[2][1]
                subj = (subj0,subj1)
            if s[0][0] in triple[2][0]:
                obj0 = word_insert(triple[2][0],s[0][0],s[2][0])
                obj1 = triple[2][1]
                if r.match(s[2][1]):
                    obj1 = s[2][1]
                obj = (obj0,obj1)
            new_triple = (subj,rel,obj)
            triple = new_triple
        new_dep_triples.insert(index,triple)
    return new_dep_triples

#fr = open(sys.argv[1], 'rt')
#text = fr.read()
#sentences = nltk.sent_tokenize(text)
#fw = open("seva-triples", 'wt')

#full_triple_array = []

def toie(sentence):

    triple_array = []
    token = nltk.word_tokenize(sentence)
    tagged_sentence = stanford_pos_tagger.tag(token)
    parsed_sentence = stanford_dependency_parser.raw_parse(sentence)
    dep = next(parsed_sentence)
    dep_triples = list(dep.triples())
    
    #resolve compound in dependecy tree
    dep_triples = joinCompoundsInDepTree(dep_triples)
    
    #print(dep_triples)
    
    #find subject-verb-object
    set_of_triples = SVO(dep_triples)
    svo_triples = None
    if not set_of_triples:
        son_triple = relWithNoObjs(dep_triples)
        if son_triple != None:
            temp_array = []
            temp_array.append(son_triple)
            set_of_triples = temp_array
    if set_of_triples != None:
        svo_triples = set_of_triples.copy()
    
    #resolve compound words
    #for i in range(len(set_of_triples)):
        #set_of_triples[i] = compound(dep_triples, set_of_triples[i])
    
    #add triples to the list
    if set_of_triples:
        triple_array.extend(set_of_triples)
    
    #IS-A relationships
    set_of_triples = IS_A(dep_triples)
    if set_of_triples:
        triple_array.extend(set_of_triples)
        
    #additional relationships
    #set_of_triples = additionalExtractions(dep_triples, tagged_sentence, svo_triples)
    #if set_of_triples:
        #triple_array.extend(set_of_triples)
        
    # Extraction for amod, nmod - using dep tree
    set_of_triples = traverseDepTree(dep_triples)
    if set_of_triples:
        triple_array.extend(set_of_triples)

            
    #resolve compound words
    for i in range(len(triple_array)):
        triple_array[i] = compound(dep_triples, triple_array[i])

    return triple_array

    #print(sentence)
    #print(triple_array)
    #print('\n')
    
    #full_triple_array.extend(triple_array)
    #print(triple_array)

#for t in full_triple_array:
    #print(t)
    
#fw.close()
#fr.close()