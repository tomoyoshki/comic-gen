# comic-gen

**Ruipeng Han, Tomoyoshi Kimura, Kaiyuan Luo, Tianchen Wang, Weiyang Wang**

## Internal

### Links

- [Overleaf report write up](https://www.overleaf.com/read/bbhrzrgfqcst#65c342)
- [Experimental Results](https://docs.google.com/spreadsheets/d/1LtugnDpvXAg4tg7iXpLiMLymPbm5JdXk3aWUOmo4b2o/edit?usp=sharing)

### TODO

- [X] Dataset
- [X] Dataloader
  - [X] Sequential panels
  - [X] Text loader
  - [X] Text tokenizer
- [ ] Models
  - [ ] General
    - [ ] Vision Encoder
    - [X] Language Encoder
    - [X] Decoder
    - [X] Loss
  - [ ] Codigen
    - [ ] Mutual Sequential modules (Sequential modules of image + language embeddings)
    - [ ] Fusion modules
  - [ ] Baselines
    - [ ] Vision only (Vision embedding concat)
    - [X] Language only (Language embedding concat)
    - [ ] Non sequential (Vision + Language embedding concat)
    - [ ] Indpendent Sequential modules (Vision sequential or Language sequential)
    - [ ] Contrastive baseline?
    - [ ] CLIP baseline?
- [ ] Evaluations
  - [ ] Encoder evaluation metrics
  - [ ] Decoder evaluation metrics
- [ ] Write up
  - [ ] Abstract
  - [ ] Introduction
  - [ ] Related Works
  - [ ] Methodology
  - [ ] Evaluation
  - [ ] Discussion
  - [ ] Future plans
  - [ ] Appendix

## About

In the world of comics, dialogues play a pivotal role in conveying narrative and character depth. A pressing issue in the industry is the completion of dialogues in comics that contain incomplete segments, especially when the concluding block lacks the dialogue altogether. This project proposes a novel solution to this challenge: a multimodal learning model designed to generate these missing dialogues. By leveraging previous advancements in the field, we aim to introduce innovative and creative techniques, which makes use of both pictures and descriptive texts on the plots, ensuring the generated dialogues align with the character's established personality and prior conversational context. In particular, we use our sequential encoders to process both inputs, converting them into abstract representations. These are then fed into a language decoder to produce the final dialogues. At the end of this project, we anticipate delivering a comprehensive report detailing our methodologies, findings, and insights, along with a functional generative model that can be utilized to generate a sequence of persona-aware dialogues for most comics.

## Usage

**Create Conda environment**

```bash
conda create --name [env_name] python=3.10
conda activate [env_name]
```

**Update pip**

```bash
pip install --upgrade pip
```


**Clone the repo and install packages**

```
git clone [repo]
cd comic-gen
pip install -r requirements.txt
```

**Argument list**

```
python3 train.py -h
```

### Available framework and baselines

- Codigen
- LanguageNonSequential (baseline)

### Train Encoder

**Train Codigen**

```python
python3 train.py -gpu=[GPU] -framework=Codigen
```

**Train baselines**

```python
python3 train.py -gpu=[GPU] -framework=Baseline -baseline=[BASELINE TO RUN]
```

### Train Decoder

**Train Codigen Decoder**

```python
python3 train.py -gpu=[GPU] -framework=Codigen -stage=decode  -model_weight[PATH TO MODEL ENCODER WEIGHT]
```

**Train Baselines Decoder**

```python
python3 train.py -gpu=[GPU] -framework=Baseline -baseline=[BASELINE TO RUN] -stage=decode -model_weight[PATH TO MODEL ENCODER WEIGHT]
```

