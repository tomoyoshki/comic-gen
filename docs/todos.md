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



