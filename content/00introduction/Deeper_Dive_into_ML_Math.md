---
title: "Appendix - Deeper Dive into ML Math"
linkTitle: "Deeper Dive into ML Math"
weight: 4
---

## Neural Node (Neuron) Equation

\[
a = \sigma(Wx + b)
\]

This equation describes **how a single neuron produces an output**.

---

### `x` — Input(s)
- The data coming **into** the neuron.
- Each `x` is typically one feature from the dataset (e.g., pixel intensity, temperature, word embedding value).
- Can be a single value or a vector of values.

**Intuition:** *What the neuron “sees.”*

---

### `W` — Weights
- Numbers that scale each input.
- Learned during training.
- Determine how important each input is.

**Intuition:** *How much the neuron cares about each input.*

---

### `b` — Bias
- A constant added to the weighted sum.
- Allows the neuron to shift its activation left or right.
- Prevents the neuron from being forced to pass through zero.

**Intuition:** *The neuron’s built-in offset or threshold.*

---

### `Wx + b` — Weighted Sum
- The linear combination of inputs and weights plus bias.
- This is the raw, unactivated signal.

**Intuition:** *The neuron’s total input signal before making a decision.*

---

### `\sigma(\cdot)` — Activation Function
- A non-linear function (e.g., sigmoid, ReLU, tanh).
- Determines how strongly the neuron “fires.”
- Enables neural networks to model complex, non-linear patterns.

**Intuition:** *The decision rule that turns signal into output.*

---

### `a` — Activation (Output)
- The final output of the neuron.
- Passed to the next layer or used as a prediction.

**Intuition:** *What the neuron outputs to the rest of the network.*

---

### One-Sentence Summary

> A neuron multiplies inputs by learned weights, adds a bias, and passes the result through an activation function to produce an output.
