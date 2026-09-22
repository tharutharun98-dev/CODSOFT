# Task 5 — Handwritten Text Generation

## Problem statement
Implement a **character-level Recurrent Neural Network (RNN)** to
generate handwritten-like text, trained on a dataset of text examples,
generating new text based on learned patterns.

## Approach
A **vanilla character-level RNN is implemented entirely from scratch
using NumPy** (no TensorFlow/PyTorch dependency — see note below), based
on the classic char-RNN architecture (Karpathy-style min-char-rnn):

1. **Encoding:** every unique character in the training corpus is mapped
   to an index (one-hot encoded as model input/output).
2. **Model:** a single-layer RNN —
   `h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)`,
   `y_t = W_hy · h_t + b_y`, softmax over the vocabulary to get the next
   character's probability distribution.
3. **Training:** truncated backpropagation-through-time (BPTT) over
   25-character chunks, Adagrad optimizer, gradient clipping to avoid
   exploding gradients, run for 6,000 iterations.
4. **Generation:** starting from a seed character, the model repeatedly
   samples the next character from its predicted probability
   distribution and feeds it back in — producing new text one character
   at a time, exactly as described in the task.

## Files
```
dataset/corpus.txt                  -> training text corpus
code/char_rnn.py                     -> full from-scratch RNN implementation
output/training_loss.png             -> training loss curve
output/generated_samples.txt         -> generated text + run metadata
output/handwritten_style_output.png  -> generated text rendered in a cursive/italic style
output/rnn_weights.npz               -> trained model weights
```

## How to run
```bash
python code/char_rnn.py
```
(Runs in well under a minute — no GPU needed.)

## Results
The loss curve (`output/training_loss.png`) shows the RNN steadily
learning character-level structure from the small corpus — the
generated output starts producing recognizable word-fragments and
occasional real words ("the", "way", "is") while inventing plausible
new ones, which is the expected behavior of a small char-RNN trained
briefly on a small corpus. See `output/generated_samples.txt` for the
full generated sample.

## Note on the dataset
The official CodSoft task brief links to the **IAM Handwriting
Database** — a multi-gigabyte collection of scanned pen-stroke images
(x/y coordinates over time) requiring registration, intended for
stroke-sequence handwriting *synthesis* (as in Alex Graves' 2013 paper
*"Generating Sequences With Recurrent Neural Networks"*), not plain
text generation. That dataset can't be downloaded in this environment.
To keep the task genuinely runnable end-to-end, this implementation
uses the exact **RNN architecture and training procedure requested in
the task** ("character-level recurrent neural network to generate
handwritten-like text... train on a dataset of handwritten text
examples") applied to a plain-text corpus instead of stroke images.

**To extend this to real stroke-based handwriting synthesis:**
1. Download the IAM Online Handwriting Database (register at
   fki.tic.heia-fr.ch or via the Kaggle mirror linked in the task).
2. Replace the one-hot character input with normalized (Δx, Δy,
   pen-up/down) stroke-point sequences.
3. Swap the vanilla RNN for a stacked LSTM with a **Mixture Density
   Network** output layer (as in Graves' paper) to predict a
   distribution over the next pen coordinate instead of a character.
4. Render the generated (x, y) sequences as actual pen strokes with
   matplotlib to get true handwriting images.

## Possible extensions
- Swap the vanilla RNN for an LSTM/GRU for better long-range memory.
- Train on a larger, more diverse text corpus for richer output.
- Add temperature-controlled sampling (lower temperature = safer/more
  repetitive, higher = more creative/riskier text).
