# Custom Transformer Model

This project implements a custom transformer model with global and local attention mechanisms. It is designed for language modeling tasks and incorporates dilated hierarchical convolutions and a shared static tensor pool.

## Features

- **GlobalLocalAttention:** Combines global attention (using a static tensor pool and attention router) with local attention (using dilated hierarchical convolutions).
- **Dilated Hierarchical Convolutions:** Captures local context using multiple layers of dilated convolutions.
- **Shared Static Tensor Pool:** Provides a memory component for the global attention mechanism.
- **Custom Transformer Block:** Integrates the GlobalLocalAttention mechanism into a transformer block.
- **Streaming Dataset:** Uses the `datasets` library to load and process data in streaming mode, enabling training on large datasets.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Transformers library
- Datasets library

To install the requirements, run:

```bash
pip install torch transformers datasets
```

## Usage

### 1. Clone the repository:

```bash
git clone https://github.com/GetnCode/llm.git
cd llm
```

### 2. Prepare the dataset:

The training script uses the wikitext-103 dataset from the `datasets` library. No manual download is required, as the dataset is loaded in streaming mode during training.

### 3. Run the training script:

```bash
python train.py
```

### Configuration

The model and training configurations are defined in the `train.py` file. You can adjust parameters such as:

- `hidden_dim`: Hidden dimension of the transformer model.
- `num_layers`: Number of transformer layers.
- `max_seq_len`: Maximum sequence length.
- `batch_size`: Training batch size.
- `learning_rate`: Learning rate for the optimizer.
- `num_epochs`: Number of training epochs.

### Checkpoints

The training script saves checkpoints periodically to the `checkpoint_lm_model` directory. You can use these checkpoints to resume training or evaluate the model.

## Model Architecture

The core of the model is the `GlobalLocalAttention` module, which combines local and global context. The local context is captured using dilated hierarchical convolutions, while the global context is captured using a shared static tensor pool and an attention router. The `CustomTransformerBlock` integrates this attention mechanism into a standard transformer block.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.
