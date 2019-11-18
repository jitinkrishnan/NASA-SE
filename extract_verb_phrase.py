import nltk
from nltk.chunk import RegexpParser

def extract_vp(pos_tagged_phrase):

    # basic check on the first and last word of the phrase
    special_chars = [".", ";", "!", ",", ":", "/"]
    if pos_tagged_phrase[0][0] in special_chars or pos_tagged_phrase[-1][0] in special_chars:
        return [],[]

    phrases = []
    #grammar = r"""
            #VP: {(<MD>|<R.*>|<I.*>)?<VB.*>+(<.*>?(<VB.*>|<JJ.*>|<R.*>|<I.*>))?<TO>?<VB.*>*(<VB.*>|<JJ.*>|<R.*>|<I.*>)*}           
            #"""
    grammar = r"""
            VP: {(<MD>|<R.*>|<I.*>|<VB.*>|<JJ.*>|<TO>)*<VB.*>+(<MD>|<R.*>|<I.*>|<VB.*>|<JJ.*>|<TO>)*}           
            """
    chunk_parser = nltk.RegexpParser(grammar)
    chunk = chunk_parser.parse(pos_tagged_phrase)
    skip = False
    for subtree in chunk.subtrees():
        if skip:
            skip = False
            continue
        if subtree.label() == 'VP':
            p = []
            for wt in subtree:
                if type(wt) == nltk.tree.Tree:
                    skip = True
                    for t in wt:
                        p.append(t)
                else:
                    p.append(wt)
            phrases.append((subtree.label(), p))

    p_phrases = []
    for p in phrases:
        pos = p[1]
        p_phrases.append(" ".join([t[0] for t in pos]).strip())
    return phrases, p_phrases



