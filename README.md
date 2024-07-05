
# Transformer from Scratch (PyTorch Lightning)

## Overview

Provide a brief introduction to your project, its purpose, and its relevance. Mention how it relates to the concepts introduced in the paper "Attention is All You Need" by Vaswani et al.

## Table of Contents

1. **Introduction**
2. **Background**
3. **Architecture Overview**
4. **Implementation Details**
5. **Usage**
6. **Results**
7. **Conclusion**
8. **References**

## 1. Introduction

Explain the motivation behind your project and the problem you are addressing. Mention the significance of the Transformer model and its relevance in natural language processing and other domains.

## 2. Background

### 2.1 Transformer Model

Provide a detailed explanation of the Transformer model as described in the paper "Attention is All You Need". Cover the following points:

- **Self-Attention Mechanism:** Explain how self-attention works and its advantages over traditional recurrent and convolutional networks.
- **Multi-Head Attention:** Describe the concept of multi-head attention and its role in enhancing the model's ability to focus on different aspects of the input sequence.
- **Positional Encoding:** Discuss how positional encoding is used to inject positional information into the input sequences.

## 3. Architecture Overview

### 3.1 Model Architecture

- **Encoder:** Explain the structure of the encoder, including the stacking of encoder layers, multi-head attention, and feed-forward networks.
- **Decoder:** Detail the structure of the decoder, focusing on the decoder layers, multi-head attention (both self-attention and cross-attention), and feed-forward networks.
- **Input Embeddings:** Discuss how input embeddings are utilized for both source and target sequences.
- **Positional Encoding:** Explain the method used for positional encoding in your implementation.

## 4. Implementation Details

### 4.1 Training Process

- **Loss Function:** Cross Entropy Loss was used with `ignore_index` and `smoothing`.

~~~pyhton
loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
~~~

- **Optimizer:** Adam was used with learining rate 10**-4.

~~~python
optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], eps=1e-9
        )
~~~

- **Learning Rate Schedule:** No learning rate scheduler was used.

## 5. English to Hinglish Translation Model

### Dataset

The model was trained using the `findnitai/english-to-hinglish` dataset available on Hugging Face. This dataset consists of English sentences paired with their Hinglish translations, facilitating the training of a translation model between these languages.

### Model Training

The model architecture is based on the Transformer model as described in the paper "Attention is All You Need" by Vaswani et al. The implementation leverages PyTorch and PyTorch Lightning for efficient training and experimentation. The model was trained using a batch size of 10, for 20 epochs, with a learning rate of 0.0001, and a sequence length of 300 tokens.

### Performance Metrics

During training, the model's performance was evaluated using metrics such as BLEU score, Word Error Rate (WER), and Character Error Rate (CER) to assess translation quality.

## 6. Results

- **Quantitative Results:** Provide metrics such as BLEU score, Word Error Rate, or any other relevant evaluation metrics.
- **Qualitative Results:** Include sample outputs or predictions generated by your model.

## 7. References

- **Paper**: "Attention is All You Need" by Vaswani et al., 2017.
- **Dataset**: `findnitai/english-to-hinglish` dataset on Hugging Face.

---

<!-- ### Notes for Pictures:

- **Visualizing Attention Mechanism:** Include diagrams showing how attention weights are computed.
- **Transformer Architecture:** Diagrams illustrating the overall architecture of the Transformer model.
- **Training Process:** Charts or graphs depicting training/validation loss over epochs or other relevant metrics. -->