---
title: "Build the Classifier"
linkTitle: "Build the Classifier"
weight: 3
---

## Building a 10-Dimensional Transformer Spam Classifier

In this chapter, you will combine everything you've learned to build a full Transformer-based text classifier.
This model will take raw text messages as input and predict whether the message is spam or not spam (ham).

You will see how to connect the data pipeline, embeddings, Transformer block, and final classifier into one coherent model.

### Define the Model Hyperparameters
These are hyperparameters — settings that control the architecture of your model.
- embed_dim → size of each token vector
- num_heads → number of attention heads
- ff_dim → internal size of the feed-forward network inside the Transformer
- dropout_rate → percent of units randomly dropped during training

#### Why do we define them here?

To make the model easy to tune. Students can experiment with different values to see how they affect:
- training stability
- model accuracy
- model capacity

These hyperparameters do not change during training; they shape the architecture.

```python
embed_dim = 10   # requirement: 10-dimensional representation
num_heads = 2    # can be tuned (must divide embed_dim)
ff_dim    = 8   # feed-forward size (students can tune)
dropout_rate = 0.7
```

### Create the Input Layer for Raw Text

```python
text_input = layers.Input(shape=(), dtype=tf.string, name="text")
```

This is the entry point of your model. It tells Keras:
- the input type → string
- the input shape → a single text field

#### Why is this important?
Transformers normally operate on numeric sequences (token IDs), so we need a layer that accepts raw text before any processing.

### Convert Text Into Token IDs
```python
x = vectorize_layer(text_input)
```
The vectorize_layer maps text into integer token IDs.

```python
"Hello world" → [17, 45]
```
#### Why do we do this?

Neural networks cannot operate on text directly.
They only work with numbers.
Vectorization is the first preprocessing step of any NLP model.

### Embed the Token IDs Into 10-Dimensional Vectors

An embedding layer converts token IDs into dense vectors of length `embed_dim`.

If `embed_dim = 10`, then each token becomes a 10-dimensional representation.

```python
x = layers.Embedding(
    input_dim=max_tokens,
    output_dim=embed_dim,
    name="token_embedding"
)(x)
```
#### Why do we need embeddings?

Because:
- token IDs are arbitrary numbers
- embeddings allow the model to learn semantic meaning
- similar words get similar vectors

This is the first learnable layer of the model.


### Apply the Transformer Block (Encoder)
You are applying the Transformer block you implemented earlier.

```python
x = TransformerBlock(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    rate=dropout_rate,
)(x)
```
#### Why do we use a Transformer here?

Transformers are exceptionally good at:
- modeling sequences
- understanding context
- learning relationships between tokens

In spam classification, word relationships matter:

- “free” → suspicious
- “free” + “win money” → very suspicious
- “free” + “trial” → maybe okay

The Transformer captures these relationships.

### Reduce the Sequence to a Single Vector
This operation takes all token embeddings from the sequence and computes one single averaged vector.

Example:
If your sequence has 100 tokens → turn this into one vector of size 10.

```python
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(dropout_rate)(x)
```

#### Why do we do this?

Because the classifier expects one vector per input, not one vector per token.
Think of it like summarizing the entire message into its most important features.
You may ask why add dropout?
To reduce overfitting and help generalization.

### Add the Output Layer for Binary Classification

We create a dense output neuron that predicts:
- 0 → ham (not spam)
- 1 → spam

```python
output = layers.Dense(1, activation="sigmoid", name="spam_score")(x)
```

### Build the Final Model
We connect:
- the input
- preprocessing
- embeddings
- Transformer
- pooling
- classifier

- into a single trainable model.

```python
model = keras.Model(inputs=text_input, outputs=output)
```

To get an overview of the final model architecture, you can print a `summary()` at the end:
```python
model.summary()
```

## Complete Code

```python
#@title Build 10-dimensional Transformer spam classifier

embed_dim = 10   # requirement: 10-dimensional representation
num_heads = 2    # can be tuned (must divide embed_dim)
ff_dim    = 8   # feed-forward size (students can tune)
dropout_rate = 0.7

# Input is raw text
text_input = layers.Input(shape=(), dtype=tf.string, name="text")

# Text -> token IDs
x = vectorize_layer(text_input)

# Token IDs -> embeddings (10-dimensional)
x = layers.Embedding(
    input_dim=max_tokens,
    output_dim=embed_dim,
    name="token_embedding"
)(x)

# Transformer encoder
x = TransformerBlock(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    rate=dropout_rate,
)(x)

# Reduce sequence to a single vector
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(dropout_rate)(x)

# Output layer: spam (1) vs ham (0)
output = layers.Dense(1, activation="sigmoid", name="spam_score")(x)

model = keras.Model(inputs=text_input, outputs=output)

model.summary()
```