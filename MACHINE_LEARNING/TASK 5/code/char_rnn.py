"""
char_rnn.py
------------
TASK 5: HANDWRITTEN TEXT GENERATION

A character-level Recurrent Neural Network (vanilla RNN), implemented
from scratch with NumPy (no deep-learning framework dependency), that
learns character sequences from a small text corpus and generates new,
"handwritten-style" text one character at a time based on learned
patterns -- i.e. the model does not copy phrases, it predicts the next
character from the previous hidden state + character, exactly as
described in the task brief.

NOTE ON THE DATASET: the official CodSoft link points to the IAM
Handwriting dataset (images of real handwriting strokes), which is a
multi-gigabyte dataset requiring registration and is meant for
image/stroke-based handwriting synthesis (e.g. with LSTMs over pen
stroke coordinates, as in Alex Graves' "Generating Sequences With
Recurrent Neural Networks"). That is not fetchable in this sandboxed
environment. To keep the task genuinely runnable end-to-end here, this
script instead implements the CHARACTER-LEVEL RNN part of that idea
(the part explicitly asked for in the task: "character-level recurrent
neural network (RNN) to generate handwritten-like text") using a plain
text corpus as training data. The same RNN architecture generalizes
directly to stroke-sequence data if the real IAM dataset is used later
(see README.md for how to swap it in).

Run:
    python code/char_rnn.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "corpus.txt")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------
with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_size = len(text)
print(f"Corpus has {data_size} characters, {vocab_size} unique characters.")

# ---------------------------------------------------------------
# 2. Hyperparameters & model parameters (vanilla RNN)
# ---------------------------------------------------------------
hidden_size = 128
seq_length = 25
learning_rate = 0.1
n_iterations = 6000

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01   # input -> hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden -> hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01   # hidden -> output
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def loss_fn(inputs, targets, hprev):
    """Forward + backward pass through the RNN for one training chunk."""
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)
        ys[t] = Why @ hs[t] + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0] + 1e-12)

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += dy @ hs[t].T
        dby += dy
        dh = Why.T @ dy + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t - 1].T
        dhnext = Whh.T @ dhraw

    for dparam in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """Generate n characters starting from seed_ix and hidden state h."""
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for _ in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# ---------------------------------------------------------------
# 3. Training loop (Adagrad)
# ---------------------------------------------------------------
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

n, p = 0, 0
smooth_loss = -np.log(1.0 / vocab_size) * seq_length
loss_history = []

while n < n_iterations:
    if p + seq_length + 1 >= len(text) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in text[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in text[p + 1:p + seq_length + 1]]

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fn(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 200 == 0:
        loss_history.append((n, smooth_loss))
        print(f"iter {n:5d}  loss {smooth_loss:.4f}")

    for param, dparam, mem in zip(
        (Wxh, Whh, Why, bh, by), (dWxh, dWhh, dWhy, dbh, dby), (mWxh, mWhh, mWhy, mbh, mby)
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1

# ---------------------------------------------------------------
# 4. Generate handwritten-style text samples
# ---------------------------------------------------------------
seed = char_to_ix[text[0]]
hstate = np.zeros((hidden_size, 1))
generated_ixes = sample(hstate, seed, 400)
generated_text = "".join(ix_to_char[ix] for ix in generated_ixes)

print("\n----- Generated text sample -----")
print(generated_text)
print("----------------------------------")

with open(os.path.join(OUT_DIR, "generated_samples.txt"), "w") as f:
    f.write("TASK 5: HANDWRITTEN TEXT GENERATION - MODEL OUTPUT\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Training corpus size: {data_size} characters, vocab size: {vocab_size}\n")
    f.write(f"Final smoothed loss after {n_iterations} iterations: {smooth_loss:.4f}\n\n")
    f.write("Generated sample (400 characters), sampled character-by-character\n")
    f.write("from the trained RNN's learned probability distribution:\n\n")
    f.write(generated_text + "\n")

# ---------------------------------------------------------------
# 5. Plot training loss curve
# ---------------------------------------------------------------
iters, losses = zip(*loss_history)
plt.figure(figsize=(8, 5))
plt.plot(iters, losses, color="#4C72B0")
plt.xlabel("Training iteration")
plt.ylabel("Smoothed loss")
plt.title("Task 5: Char-RNN Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Render the generated text as a "handwritten-style" image
# ---------------------------------------------------------------
wrapped = []
line = ""
for ch in generated_text:
    line += ch
    if len(line) >= 55 or ch == "\n":
        wrapped.append(line)
        line = ""
if line:
    wrapped.append(line)

fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
y = 0.95
for line in wrapped[:14]:
    ax.text(0.03, y, line, fontsize=14, fontstyle="italic", family="cursive",
             transform=ax.transAxes)
    y -= 0.07
ax.set_title("Char-RNN Generated 'Handwritten-Style' Text Output", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "handwritten_style_output.png"), dpi=150)
plt.close()

# Save model weights
np.savez(os.path.join(OUT_DIR, "rnn_weights.npz"),
          Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by,
          chars=np.array(chars))

print("\nAll outputs saved to the 'output' folder.")
