## NASA-SE
A Virtual Assistant for NASA's Systems Engineers

This repo is a combination of two projects:
*   **[`SEVA: A Systems Engineer's Virtual Assistant`](http://ceur-ws.org/Vol-2350/paper3.pdf)** (AAAI-MAKE 2019)
*   `Common Knowledge Concept Recognition for SEVA` (In progress)

Datasets used in the project are availale in the [datafolder](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data). 

### 1. SEVA-TOIE
SEVA-TOIE is a targetted open domain information extractor for simple systems engineering sentences which is based on domain specific rules constructed over universal dependencies. It extracts fine-grained triples from sentences and can be used for downstream tasks such as knowledge graph construction and question-asnwering.

#### How to run:

```pip install -r requirements.txt``` to install necessary packages (if needed).

```
cd NASA-SE
python -i seva_toie.py
>>> sentence = "STI is an instrument."
>>> toie(sentence)
>>> [('STI', 'is', 'instrument)]
>>> sentence = "STI, an instrument, has a 2500 pixel CCD detector."
>>> toie(sentence)
>>> [('STI', 'has', 'CCD detector'), ('STI', 'is-a', 'instrument'), ('CCD detector', 'has-property', '2500 pixel')]
```
Try with more example/template [sentences](https://github.com/jitinkrishnan/NASA-SE/blob/master/se_data/seva-toie-sentences.txt).

### 2. Concept Recognition (CR)

#### Training and Evaluating a custom CR model

##### Update Vocabulary

##### Train and Evaluate

#### Running the model on an input sentence

#### Construct a Knowledge Graph

#### Verb Phrase Chunking

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
