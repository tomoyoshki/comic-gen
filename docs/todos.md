# ToDo (10/31/23)

## Datasets (1 member)

1. Download datasets
2. Integrate datasets into the code
    - `multi_modal_dataset.py`
    - Need to have (N images and N texts per sample, N images and N - 1 texts are used for models, and the last text is used as label to compare with model output)

## Models todo (2 - 3 members)

### Proposed frameworks

1. organize ideas and decide architecture
2. implement the architecture
3. implement the loss function

### Baselines

1. Decide baselines to run
2. implement baselines
3. Implement baseline losses


#### Potential Baselines

1. CMC?
2. Direct Similarity?
....


## Evaluation metrics (1 member)

1. Decide what evaluation metrics to use
2. Similarity between the generated output and the ground truth?
3. Implement them in code (`train_utils/evaluations.py`)





## 11/2

### Task 1 (Ruipeng + Tim)

1. Find pretrained model weight for vision transformer
2. Find appropriate model for Language encoder
3. Find pretrained model weight for language model

### Task 2 (Ruipeng + Tim)

1. Codigen framework implementation
2. Decide on fusion and sequential network 


### Task 3 

1. Baselines (2 - 3)

### Task 4

1. Decoders? 
2. Metrics between decoded text and ground truth text, (angram? how gpt works?)

### Task 5 (tommy)

1. Pipelines, test, evaluations
2. Loss implementation (encoder + contrasitve/similarity)