---
title: "Creating the Transformer"
linkTitle: "Creating the Transformer"
weight: 2
---

## Building the Transformer Block
In this part of the workshop, you will construct a complete Transformer block from scratch using Keras.
Transformers are the foundation of modern AI systems such as BERT, GPT, and Vision Transformers.
By implementing the block yourself, you’ll understand how these models actually work under the hood.

We will build the block piece by piece and discuss what each component does and why we need it.

### Start by Creating a Custom Layer
We begin by defining a new layer class called TransformerBlock.
Using a custom layer gives us a reusable building block that we can stack when creating our model later.
A custom layer is your own building block inside a neural network.
Keras has built-in layers (Dense, Conv2D, etc.), but sometimes you need functionality that doesn't exist yet.
Transformers require a specific structure, so we create a reusable building block.

Why do we do this?
- To encapsulate all Transformer logic in one place
- To reuse it multiple times
- To allow saving/loading the model later 
- To make the code cleaner and modular

```python
class TransformerBlock(layers.Layer):
```

Inside this class, we will define all parts that make up a standard Transformer block.

### Add Hyperparameters in the Constructor

Hyperparameters are settings you choose before training a model.
They control the architecture and behavior of the layer but are not learned during training.
Examples:
- embedding dimension
- number of attention heads
- dropout rate

**Why are hyperparameters important?**
They determine:
- model capacity
- complexity
- memory usage
- training stability

Choosing good hyperparameters can drastically improve performance.

Next, we prepare the configuration of the block in the __init__ constructor:

```python
def __init__(self, embed_dim=10, num_heads=2, ff_dim=32, rate=0.1, **kwargs):
    super().__init__(**kwargs)
```

Here you introduce the core parameters:
- `embed_dim` – dimensionality of each input vector
- `num_heads` – number of attention heads
- `ff_dim` – hidden layer size in the feed-forward network
- `rate` – dropout rate to reduce overfitting

Passing `**kwargs` to `super() ensures Keras can handle things like serialization.

### Add the Multi-Head Self-Attention Layer

Transformers rely heavily on self-attention, where each token learns which other tokens are important.
Self-attention lets each input position (token) look at all other positions and decide which ones matter.
Example:
In the sentence: `The cat sat on the mat` the word “cat” may pay more attention to “sat” than to “mat”.

#### What is multi-head attention?
Instead of having one attention mechanism, we have multiple heads.
Each head can learn different relationships.

#### Why is this layer here?
Self-attention is the core mechanism that makes Transformers powerful.
It allows the model to learn context and relationships without convolution or recurrence.

Add the attention layer:
```pathon
self.att = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=embed_dim // num_heads,
)
```
Multi-head attention lets the model look at the input from several “perspectives” at once.
This is one of the reasons why Transformers are so effective.

### Build the Feed-Forward Network (FFN)
After attention, every Transformer block uses a small two-layer neural network:
A tiny two-layer neural network that is applied to every token independently after attention.
Self-attention mixes information between tokens, but the model still needs a non-linear transformation to increase expressiveness.

The FFN helps:
- refine the representation
- build complex features
- stabilize and enrich the internal structure of the model

This is part of every Transformer block.

```python
self.ffn = keras.Sequential([
    layers.Dense(ff_dim, activation="relu"),
    layers.Dense(embed_dim),
])
```
Why do we need this?
- attention gathers information
- the FFN transforms and refines that information
- this combination gives the model expressive power

### Add Layer Normalization, Dropout & Residual Connections

It normalizes the input features to stabilize training.
Transformers rely heavily on this to avoid exploding/vanishing gradients.

#### What is Dropout?

A regularization technique that randomly drops units during training to prevent overfitting.

#### What are Residual Connections?

When you see: `inputs + something` that is a skip/residual connection.

#### Why are these components here?
Transformers often fail to train without them. They:
- make gradients flow better
- improve convergence
- prevent overfitting
- stabilize deep architectures

This pattern (attention → normalization → FFN → normalization) is standard in modern Transformer models.

To make the block stable and trainable, we add:
- Layer Normalization
- Dropout
- Residual (skip) connections

```python
self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
self.dropout1 = layers.Dropout(rate)
self.dropout2 = layers.Dropout(rate)
```

Residual connections are essential—they help gradients flow through the network and prevent training from failing.


### Define the Forward Pass
Now we describe how data moves through the block. This is where everything comes together.

#### Self-Attention + Residual + Normalization

- The input attends to itself → self-attention
- Dropout is applied
- Add a residual connection: inputs + attn_output
- Normalize the result

```python
attn_output = self.att(inputs, inputs)
attn_output = self.dropout1(attn_output, training=training)
out1 = self.layernorm1(inputs + attn_output)
```
The input attends to itself
- Dropout is applied
- We add the residual connection: inputs + attn_output
- Then normalize the result


#### Feed-Forward + Residual + Normalization

- Apply the FFN
- Dropout again
- Add another residual connection
- Apply the final normalization

```python
ffn_output = self.ffn(out1)
ffn_output = self.dropout2(ffn_output, training=training)
return self.layernorm2(out1 + ffn_output)
```
You’ll notice the same pattern:
1.transformation
2.residual connection
3.normalization

This two-step structure is the core of every Transformer block.

### Make the Layer Serializable
Finally, we add a `get_config()` method so Keras can save and reload the layer. Serialization means saving your model to disk so you can reload it later.
```python
def get_config(self):
    config = super().get_config()
    config.update({
        "embed_dim": self.embed_dim,
        "num_heads": self.num_heads,
        "ff_dim": self.ff_dim,
        "rate": self.rate,
    })
    return config
```

This is required if you want to export the model or deploy it later.

### Complete Code
```python
#@title Define Transformer block

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=10, num_heads=2, ff_dim=8, rate=0.7, **kwargs):
        # IMPORTANT: pass **kwargs to super() to accept 'trainable', 'dtype', etc.
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # key_dim is embed_dim // num_heads -> here 5
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        # Let Keras know how to serialize our custom args
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config
```