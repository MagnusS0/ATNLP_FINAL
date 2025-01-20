# ATNLP-Project: Transformer-Based SCAN Task

This repository is a reimplementation of experiments from the paper [*Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks*](https://arxiv.org/abs/1711.00350) by Brendan Lake and Marco Baroni. Instead of using RNNs, GRUs, or LSTMs as in the original paper, we implement a Transformer-based model inspired by the architecture proposed in the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al. Additionally, we explore the performance of a modern decoder-only architecture by fine-tuning Llama 3.2 on the same task.


## Introduction
The goal of this project is to evaluate the compositional generalization capabilities of Transformer models on the SCAN dataset. SCAN is a synthetic dataset that pairs commands (e.g., "walk twice and jump") with corresponding actions (e.g., "WALK WALK JUMP"). We test the ability of our models to generalize across three key splits:

### Transformer Experiments
- **Experiment 1**: Simple split with varying training data sizes.
- **Experiment 2**: Length-based split.
- **Experiment 3**: Compositional split.

### Llama Experiments
We also investigate how a modern decoder-only architecture performs on the SCAN task by fine-tuning Llama 3.2 using LoRA (Low-Rank Adaptation). This allows us to compare:
- Traditional seq2seq Transformer vs. decoder-only architecture
- Impact of pre-training on compositional generalization
- Efficiency of parameter-efficient fine-tuning for this task

This project is designed for educational purposes.

The Transformer model used in this repository follows the implementation done by Vaswani et al., incorporating multi-head self-attention, positional encodings, and feed-forward layers.


## Dependencies
The project is implemented in Python using PyTorch. To get started use Poetry to install the needed packages.

## Quick Start

1. **Setup Environment**:
```bash
# Install poetry if you haven't already
pip install poetry

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

2. **Run Experiments**:
```bash
# Run specific experiments (e.g., experiment 3)
python -m experiments.train_exp_3
```


## Code Structure
This repository contains the following components:

```
src/
├── data/
│   ├── dataset.py         # Custom data loader for the SCAN dataset
│   └── dataset_llama.py   # Data loader for Llama model fine-tuning
├── model/
│   └── transformer.py     # Transformer model implementation
├── utils/
│   └── utils.py          # Greedy and Oracle decoding utilities
├── experiments/          # Training scripts for experiments
├── train.py             # Main training and evaluation script
└── train_llama.py       # Llama model fine-tuning script

data/                    # Directory for SCAN dataset files
llama/                   # Llama-specific scripts
├── llama_test.py        # VLLM inference script
├── eval.py             # Evaluation metrics calculation
└── save_model.py       # Convert and save model for VLLM
```


## Data Input
The SCAN dataset is sourced from the [SCAN repository](https://github.com/brendenlake/SCAN). Each dataset split consists of text files with lines in the format:
```
IN: <COMMAND> OUT: <ACTION>
```
Example:
```
IN: jump thrice OUT: JUMP JUMP JUMP
IN: turn left twice OUT: LTURN LTURN
```

The dataset is tokenized, and both source (commands) and target (actions) vocabularies are built dynamically. Special tokens such as `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` are used for training and evaluation.


## Usage
1. **Clone the repository**:
    ```bash
    git clone https://github.com/frasalute/ATNLP-Project.git
    cd ATNLP-Project
    ```

2. **Setup the environment**:
    ```bash
    # Install poetry if you haven't already
    pip install poetry

    # Install dependencies
    poetry install

    # Activate the virtual environment
    poetry shell
    ```

3. **Download the SCAN dataset**:
    Place the dataset files in the `data/` directory. Example structure:
    ```
    data/
      length_split/
        tasks_train_length.txt
        tasks_test_length.txt
      simple_split/
        size_variations/
          tasks_train_simple_p1.txt
          tasks_test_simple_p1.txt
          ...
    ```

4. **Run Transformer experiments**:
    ```bash
    # Run experiment 1 (simple split with varying training data sizes)
    python -m src.experiments.train_exp_1

    # Run experiment 2 (length-based split)
    python -m src.experiments.train_exp_2

    # Run experiment 3 (compositional split)
    python -m src.experiments.train_exp_3
    ```

5. **Run Llama experiments** (Optional):
    We also provide experiments using a decoder-only Llama 3.2 model with LoRA fine-tuning. These experiments explore how a large language model performs on the SCAN task compared to the Transformer model.

    ```bash
    # Step 1: Fine-tune Llama model (this will save checkpoints to outputs/)
    python -m src.train_llama

    # Step 2: Save the model in 16-bit format for VLLM inference
    # Replace [checkpoint] with the checkpoint directory (e.g., outputs/checkpoint-820)
    # Replace [output] with desired output path (e.g., outputs/model_p16_20e)
    python -m llama.save_model --checkpoint [checkpoint] --output [output]
    
    # Step 3: Run inference using VLLM for batched prediction
    # Make sure to update the model path in llama_test.py to match your saved model
    python -m llama.llama_test

    # Step 4: Evaluate the results
    python -m llama.eval
    ```

    The Llama experiments will produce:
    - Training metrics and loss curves in TensorBoard
    - A CSV file with model predictions
    - Detailed evaluation metrics including:
        - Token and sequence accuracy
        - Length-based performance analysis
        - Error examples and edit distances


## Evaluation
### Metrics
The following metrics are used to evaluate model performance:
- **Token-level accuracy**: Measures the percentage of correct token predictions.
- **Sequence-level accuracy**: Measures whether the entire output sequence matches the target sequence.

### Example Output
Each experiment produces both terminal output and visualization plots:

#### Terminal Output
```plaintext
Starting run 1/5 with seed 42
======================================================================

Training dataset size p1
Dataset p_1 - Epoch: 1/20
Train Loss: 2.3456
Test Loss: 2.1234
Teacher Forcing - Token Acc: 0.4567, Seq Acc: 0.2345
--------------------------------------------------

Final Results Summary:
==================================================
Dataset Size | Mean Accuracy ± Std Dev
--------------------------------------------------
p1          | Mean Accuracy: 0.8234 ± 0.0123
Individual runs: 0.8245, 0.8212, 0.8256, 0.8198, 0.8259
Mean Greedy Accuracy: 0.7123 ± 0.0234
Individual runs: 0.7145, 0.7089, 0.7156, 0.7102, 0.7123
```

## Results
We aim to reproduce and compare the following key findings:
- Performance of Transformers across dataset splits and sizes.
- Ability of Transformers to generalize to longer or unseen command sequences.


## Acknowledgments
This project is inspired by the experiments conducted in [Lake & Baroni, 2017](https://arxiv.org/abs/1711.00350). The Transformer model architecture is adapted from the foundational paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al.
