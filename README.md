# comic-gen

**Ruipeng Han, Tomoyoshi Kimura, Kaiyuan Luo, Tianchen Wang, Weiyang Wang**

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

