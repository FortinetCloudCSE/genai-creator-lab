---
title: "Vectorize the Training Data"
linkTitle: "Vectorize the Training Data"
weight: 3
---

## Text Vectorization

Modern neural networks cannot work directly with raw text. They require numerical input—typically sequences of integers.
This section explains how text vectorization works, why it is necessary, and how the provided code implements it.

Text vectorization is the process of converting raw text (words, sentences, documents) into numerical representations that machine-learning models can understand.

Neural networks operate using mathematical operations on vectors and matrices.
Because text is not inherently numeric, we need a way to map text → numbers in a consistent and meaningful way.

### Why Do We Need Vectorization?
1. Neural networks require fixed-size numeric tensors
Text varies in length and contains symbolic words, so we must convert it into:
- numbers
- fixed-length sequences
- uniformly structured inputs
2. Words must be represented consistently
Each word must map to the same numeric ID every time.

Example:
Word | Token ID
------|------
“movie” | 85
“great” | 12
“terrible” | 313

This allows models to learn patterns like:
“bad words → negative sentiment”

3. Vocabulary control
Datasets may contain tens or hundreds of thousands of distinct words.
Vectorization allows us to choose:
- how many words to keep
- how to handle unknown words (OOV tokens)
- how long each sequence should be

4. Efficient training
A clean numeric representation reduces memory usage and simplifies model operations.

### How Does Vectorization Work Internally?
When using TextVectorization with output_mode="int", the process looks like this:

1. Tokenization
Text is split into tokens (usually words or subwords).
```
"This movie was amazing"  
→ ["this", "movie", "was", "amazing"]
```

2. Vocabulary Building: Turning Words into IDs

Machine learning models cannot work directly with text.  Before training begins, words must be converted into numbers.

During the `.adapt()` step, the text vectorization layer builds a **vocabulary**, which is simply a dictionary that maps words to integer IDs.

- scans all training texts
- counts how often each word appears
- keeps the most frequent words (up to max_tokens)
- assigns each word a unique integer ID

At this stage, words are treated as **symbols**, not meanings.  The model has not learned anything yet—it is only creating a lookup table.

- Example vocabulary:

{"<PAD>":0, "<OOV>":1, "the":2, "movie":3, "was":4, "good":5, ...}
```
"<PAD>" is used to pad sequences so all inputs have the same length
"<OOV>" (out-of-vocabulary) represents words not seen during training
```
3. Text → Sequence conversion

["this", "movie", "was", "amazing"]
→ [14, 3, 4, 112]
```

4. Padding or truncation
All sequences are resized to `max_len`.

#### Defining Hyperparameters
```python
max_tokens = 20000  # vocabulary size
max_len    = 200    # sequence length
```
- `max_tokens`
    - Limits vocabulary to the 20,000 most frequent words.
    - Rare words are replaced with <OOV> token ID.

- `max_len`
  - Every sequence will be exactly 200 tokens, padded or truncated accordingly.
This ensures that the neural network receives input tensors of consistent shape:
```text
(batch_size, max_len)
```

#### Creating the Vectorization Layer
```python
vectorize_layer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
)
```
This layer will eventually turn each text sample into a fixed-length integer sequence, ready for embedding or model input.

#### Adapting the Layer
```python
text_ds_for_adapt = tf.data.Dataset.from_tensor_slices(train_texts).batch(512)
vectorize_layer.adapt(text_ds_for_adapt)
```
You may ask yourself: Why do I need the adaption?
Because the layer needs to:
- build the vocabulary
- compute token frequencies
- map words → token IDs

This must be done only on training data to avoid information leakage.

As a summary, this is the complete code to vectorize the Enron Spam dataset for our machine learning model.

```python
#@title Text vectorization

max_tokens = 20000  # vocabulary size
max_len    = 200    # sequence length

vectorize_layer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
)

# Adapt vectorizer on training texts
text_ds_for_adapt = tf.data.Dataset.from_tensor_slices(train_texts).batch(512)
vectorize_layer.adapt(text_ds_for_adapt)

# Quick test
sample = train_texts[0]
print("Raw text:", sample[:200])
print("Token IDs:", vectorize_layer(tf.constant([sample]))[0][:30])
```
### Interpreting the Vectorization Output

After calling `adapt()`, the `TextVectorization` layer has built a vocabulary and can now convert raw text into numeric sequences.

Here is what the output means:

#### Raw text
```text
any software just for 15 $ - 99 $ understanding oem software
lead me not into temptation ; i can find the way myself .
# 3533 . the law disregards trifles .
```

```c
[   46   241   127     8   166   311  1447  1733   241  1156    47    31
   123 18000    15    43   299     2   248  1503     1     2  1094     1
     1     0     0     0     0     0]
```
This array is the **numerical representation of the text**, where:

- Each number corresponds to a word’s **ID in the vocabulary** and matches its exact position
  - i.e. any = 46, software = 241, just = 127, etc. 
- The same word always maps to the same number  
  - For example, `"software"` appears twice and maps to the same ID (241) both times
- The value `1` represents **`<OOV>` (out-of-vocabulary)**  
  - These are words not seen often enough (or at all) during training
- The value `0` represents **`<PAD>`**  
  - Padding is added at the end so all sequences have the same length
