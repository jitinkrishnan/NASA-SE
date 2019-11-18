## NASA-SE


### A Virtual Assistant for NASA's Systems Engineers

Projects in this repo:
*   `Common-Knowledge Concept Recognition for SEVA` (**In Progress**)
*   **[`SEVA: A Systems Engineer's Virtual Assistant`](http://ceur-ws.org/Vol-2350/paper3.pdf)** (AAAI-MAKE 2019)

Datasets used in the project are availale in the [datafolder](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data). 

```pip install -r requirements.txt``` to install necessary packages if needed.

### 1. Concept Recognition (CR)


#### If you just want to extract concepts right away

**Input**: A sentence.
**Output**: Tuples of concepts and their corresponding BIO labels.
```
cd NASA-SE
python -i tag_sentence.py
>>> sentence = "Acceptable Risk is the risk that is understood and agreed to by the program/project,
governing authority, mission directorate, and other customer(s) such that no further specific 
mitigating action is required."
>>> sentence2tags_all(sentence)
[('Acceptable Risk', 'mea'), ('mission', 'seterm'), ('risk', 'mea'), ('program', 'opcon'), 
('project', 'seterm'), ('mission directorate', 'seterm'), ('customer', 'grp')]
```
#### Training and Evaluating a custom CR model

##### Necessary Items

###### [BERT model](https://github.com/jitinkrishnan/NASA-SE/blob/master/bert_models)
You can change the BERT vocabulary if needed. There are a few caveats to this. The number of words should remain the same and should include the BERT tokens. BERT recommends replacing the `unused` words with domain words. However, this may not always guarantee a better performance. The example shown below updates the ```vocab.txt``` file with the words from two files: accronyms and definitions.
```
cd NASA-SE
python -i seva_dataset_utils.py
>> accr_location = "se_data/acronyms.txt"
>> definition_location = "se_data/definitions.txt"
>> vocab_location = "bert_models/vocab.txt"
update_vocab(vocab_location, accr_location, definition_location)
```
[Here](https://github.com/jitinkrishnan/NASA-SE/blob/master/SPacy-CR-Example.ipynb) is an example using spaCy.

###### [Datasets](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data)
* [CR annotated dataset](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/se_ner_annotated.tsv) 
* [Accronyms dataset](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/acronyms.txt)
* [Definitions dataset](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/definitions.txt)
* [Keyword annotated dataset](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/keywords2annotate.txt)

##### Train and Evaluate
It will take a few minutes to generate the model.
```
cd NASA-SE
python train_evaluate.py
```
Once the training is finished, we can extract the tags.
```
cd NASA-SE
python -i tag_sentence.py
>>> sentence = "Acceptable Risk is the risk that is understood and agreed to by the program/project,
governing authority, mission directorate, and other customer(s) such that no further specific 
mitigating action is required."
>>> sentence2tags_all(sentence)
[('Acceptable Risk', 'mea'), ('mission', 'seterm'), ('risk', 'mea'), ('program', 'opcon'), 
('project', 'seterm'), ('mission directorate', 'seterm'), ('customer', 'grp')]
```

#### Construct a Knowledge Graph
[Here](https://github.com/jitinkrishnan/NASA-SE/blob/master/SEVA_KG_Example.ipynb) is a jupyter notebbok example of KG construction using accronyms and definitions.

![pic](https://github.com/jitinkrishnan/NASA-SE/blob/master/images/kg_example.png)

#### Verb Phrase Chunking
Makes simple verb based connection between two near-by entities.
```
cd NASA-SE
python -i tag_sentence.py
>>> sentence = "Acceptable Risk is the risk that is understood and agreed to by the program/project,
governing authority, mission directorate, and other customer(s) such that no further specific 
mitigating action is required."
>>> verb_phrase_relations(sentence)
[('Acceptable Risk [mea]', 'is', 'risk [mea]'), ('risk [mea]', 'is understood', 'program [opcon]'),
('risk [mea]', 'agreed to by', 'program [opcon]')]
```
Examples of verb phrase extraction using POS tags.
```
cd NASA-SE
python -i tag_sentence.py
>>> extract_vp([('is', 'VBZ'), ('the', 'DT')])
([('VP', [('is', 'VBZ')])], ['is'])
>>> extract_vp([('that', 'WDT'), ('is', 'VBZ'), ('understood', 'JJ'), ('and', 'CC'), ('agreed', 'VBD'),
('to', 'TO'), ('by', 'IN'), ('the', 'DT')])
([('VP', [('is', 'VBZ'), ('understood', 'JJ')]), ('VP', [('agreed', 'VBD'), ('to', 'TO'), ('by', 'IN')])],
['is understood', 'agreed to by'])
```

### 2. SEVA-TOIE

SEVA-TOIE is a targetted open domain information extractor for simple systems engineering sentences which is based on domain specific rules constructed over universal dependencies. It extracts fine-grained triples from sentences and can be used for downstream tasks such as knowledge graph construction and question-asnwering.

```
cd NASA-SE
python -i seva_toie.py
>>> sentence = "STI is an instrument."
>>> toie(sentence)
[('STI', 'is', 'instrument)]
>>> sentence = "STI, an instrument, has a 2500 pixel CCD detector."
>>> toie(sentence)
[('STI', 'has', 'CCD detector'), ('STI', 'is-a', 'instrument'), ('CCD detector', 'has-property', '2500 pixel')]
```
Try with more example/template [sentences](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/seva-toie-sentences.txt).

### Citation
```
@inproceedings{krishnan2019seva,
  title={SEVA: A Systems Engineer's Virtual Assistant},
  author={Krishnan, Jitin and Coronado, Patrick and Reed, Trevor},
  booktitle={AAAI Spring Symposium: Combining Machine Learning with Knowledge Engineering},
  year={2019}
}
```

### Contact information

For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
