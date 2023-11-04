# comic-gen

**Ruipeng Han, Tomoyoshi Kimura, Kaiyuan Luo, Tianchen Wang, Weiyang Wang**

## Internal

### Links

- [Overleaf report write up](https://www.overleaf.com/read/bbhrzrgfqcst#65c342)
- [Experimental Results](https://docs.google.com/spreadsheets/d/1LtugnDpvXAg4tg7iXpLiMLymPbm5JdXk3aWUOmo4b2o/edit?usp=sharing)

### TODO

- [X] Dataset
- [ ] Dataloader
  - [X] Sequential panels
  - [X] Text loader
  - [ ] Text tokenizer
- [ ] Models
  - [ ] General
    - [ ] Vision Encoder
    - [ ] Language Encoder
  - [ ] Codigen
    - [ ] Mutual Sequential modules (Sequential modules of image + language embeddings)
    - [ ] Fusion modules
    - [ ] Decoder
    - [X] Loss
  - [ ] Baselines
    - [ ] Vision only
    - [ ] Language only
    - [ ] Non sequential
    - [ ] Indpendent Sequential modules
    - [ ] Contrastive baseline
    - [ ] CLIP baseline
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

- Enter the source directory

```
cd src
```

- Basic arguments

```
python3 train.py -h
```

**Train**

```python
python3 train.py -gpu=[GPU] -framework=[FRAMEWORK]
```
