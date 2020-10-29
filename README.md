## NASA-SE

### A Virtual Assistant for NASA's Systems Engineers

Projects in this repo:
*   **[`Common-Knowledge Concept Recognition for SEVA`](http://ceur-ws.org/Vol-2600/paper10.pdf)** ([AAAI-MAKE 2020](https://www.aaai-make.info))
*   **[`SEVA: A Systems Engineer's Virtual Assistant`](http://ceur-ws.org/Vol-2350/paper3.pdf)** ([AAAI-MAKE 2019](https://2019.aaai-make.info/aaai-make))

Part of the Robust Software Engineering-Led Group that won the Digital Transformation [Hackathon](https://ti.arc.nasa.gov/news/RSE-DT-Award-2020/) Award for "Most Potential NASA Impact".
September 2020.

[AAAI-MAKE 2019 Presentation Slides](https://drive.google.com/file/d/15o_uVT3OqfY9g8H_sh6wxYMr_YNk49v6/view?usp=sharing) | [Demo Slides](https://drive.google.com/file/d/1Qy7XD9w9KlISu-Rpqy-ktfrBbL4ojVaA/view?usp=sharing) | [GSFC Short Talk](https://asd.gsfc.nasa.gov/conferences/ai/program/018-Krishnan_GSFC_ShortTalkV2.pdf)

### Citation
```
@inproceedings{krishnan2019seva,
  title={SEVA: A Systems Engineer's Virtual Assistant},
  author={Krishnan, Jitin and Coronado, Patrick and Reed, Trevor},
  booktitle={AAAI Spring Symposium: Combining Machine Learning with Knowledge Engineering},
  year={2019}
}

@inproceedings{krishnan2020ckcr,
  title={Common-Knowledge Concept Recognition for SEVA},
  author={Krishnan, Jitin and Coronado, Patrick and Purohit, Hemant and Rangwala, Huzefa},
  booktitle={AAAI Spring Symposium: Combining Machine Learning with Knowledge Engineering},
  year={2020}
}
```

### Note
Datasets used in the project are availale in the [datafolder](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data). 

```pip install -r requirements.txt``` to install necessary packages if needed.

### 1. Concept Recognition (CR)

We aim to extract common-knowledge concepts for the systems engineering domain. With the help of a domain expert and text processing methods, we construct a dataset annotated at the word-level by carefully defining a BIO labelling scheme to train a NER-like sequence model to recognize systems engineering concepts.


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

##### Download Uncased [BERT model](https://github.com/google-research/bert) to NASA-SE folder.
Rename the bert folder to ```bert_models```. You can change the BERT vocabulary if needed. There are a few caveats to this. The number of words should remain the same and should include the BERT tokens. BERT recommends replacing the `unused` words with domain words. However, this may not always guarantee a better performance. The example shown below updates the ```vocab.txt``` file with the words from two files: accronyms and definitions. Vocab files we used are in ```bert_items``` folder.
```
cd NASA-SE
python -i seva_dataset_utils.py
>> accr_location = "se_data/acronyms.txt"
>> definition_location = "se_data/definitions.txt"
>> vocab_location = "bert_models/vocab.txt"
update_vocab(vocab_location, accr_location, definition_location)
```
[Here](https://github.com/jitinkrishnan/NASA-SE/blob/master/SPacy-CR-Example.ipynb) is an example using spaCy.

##### Datasets
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

![pic](https://github.com/jitinkrishnan/NASA-SE/blob/master/kg_example.png)

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

#### Jar files to be downloaded:
Place the following files in ```NASA-SE/stanford_jars``` folder
* [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml): ```stanford-parser-3.9.2-models.jar```, ```stanford-parser.jar```
* [Stanford POSTagger](https://nlp.stanford.edu/software/tagger.shtml): ```english-bidirectional-distsim.tagger```, ```stanford-postagger-3.9.2.jar```

#### Sample Run
```
cd NASA-SE
python -i seva_toie.py
>>> sentence = "STI is an instrument."
>>> toie(sentence)
[('STI', 'is-a', 'instrument)]
>>> sentence = "STI, an instrument, has a 2500 pixel CCD detector."
>>> toie(sentence)
[('STI', 'has', 'CCD detector'), ('STI', 'is-a', 'instrument'), ('CCD detector', 'has-property', '2500 pixel')]
```
Try with more example/template [sentences](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/seva-toie-sentences.txt).

### Contact information

For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
