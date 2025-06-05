# Transformer-based Neural Machine Translation for Nepali-English

This repository contains a PyTorch implementation of the Transformer architecture for Nepali-English machine translation. The implementation is based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Overview

This project implements a complete Transformer model from scratch, including:
- Custom tokenization for Nepali and English
- Multi-head attention mechanism
- Positional encoding
- Encoder-decoder architecture
- Attention visualization tools

## Features

- Complete Transformer implementation in PyTorch
- Custom tokenization pipeline for Nepali and English
- Training pipeline with TensorBoard integration
- Attention visualization tools
- Translation inference system
- Support for model checkpointing and resuming training

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── model.py              # Transformer model implementation
├── dataset.py            # Dataset and data loading utilities
├── train.py             # Training script
├── translate.py         # Translation inference script
├── config.py            # Configuration settings
├── attention_visual.ipynb  # Attention visualization notebook
├── Colab_Train.ipynb    # Google Colab training notebook
├── tokenizer_ne.json    # Nepali tokenizer
├── tokenizer_en.json    # English tokenizer
└── weights/             # Directory for model checkpoints
```

## Usage

### Training

1. Prepare your dataset in the required format
2. Configure training parameters in `config.py`
3. Run training:

```bash
python train.py
```

### Translation

To translate a Nepali sentence to English:

```bash
python translate.py "मैले धेरै पल्ट सुनेकोछु"
```

### Attention Visualization

Open `attention_visual.ipynb` in Jupyter Notebook to visualize attention patterns:

```bash
jupyter notebook attention_visual.ipynb
```

## Model Architecture

The implementation follows the original Transformer architecture with:
- 6 encoder and decoder layers
- 8 attention heads
- 512-dimensional model
- Position-wise feed-forward networks
- Residual connections and layer normalization

## Training Configuration

Default training parameters:
```python
{
    "batch_size": 64,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 100,
    "d_model": 512
}
```

## Dataset

The model is trained on the "sharad461/ne-en-parallel-208k" dataset, which contains 208,000 parallel sentences. For initial training, we use 10% of the data (approximately 20,800 sentence pairs).

## Results

The model achieves competitive results on the Nepali-English translation task. Example translations:

```
SOURCE: “तिमीले मानिसहरूलाई सच्चाइसँग न्याय गर। जब तिमी केही चीज नाप्छौ अनि तौलिन्छौ निष्पक्षतापूर्वक नाप अनि तौल।
TARGET: "'You shall do no unrighteousness in judgment, in measures of length, of weight, or of quantity.
PREDICTED: "' You shall do no injustice in judgment : from that time to measure , the length of weight and of shall be three .
```

## Future Improvements

- [ ] Implement beam search for better translation quality
- [ ] Add model distillation for faster inference
- [ ] Create a FastAPI service for real-time translation
- [ ] Build a web interface with attention visualization
- [ ] Support for more language pairs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original Transformer paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- The Hugging Face team for their tokenizers library
- The PyTorch team for their excellent deep learning framework

