---
title: "Prepare the Environment"
linkTitle: "Prepare the Environment"
weight: 3
---
## Setup Google Colab

For this lab we will use [Google Colab](https://colab.google/). This allows you, to have a stable environment including powerfull hardware to train a machine learning model.
To use Google Colab, a Google Account is required (it doesn't matter if this is your coporate or private one).

1. Go to https://colab.research.google.com/
2. Login to Google Colab by selecting the `Sign In` button at the top right corner
![login_screen](./assets/Screenshot%202025-12-09%20at%2012-45-31%20Welcome%20To%20Colab%20-%20Colab.png)
3. Select `+ New notebook` in the Wizard
![new notebook](./assets/Screenshot%202025-12-09%20at%2012-58-31%20Welcome%20To%20Colab%20-%20Colab.png)
4. Now you should be able to access the Jupyter Notebook which allows to follow along with the lab.
![workspace](./assets/Screenshot%202025-12-09%20at%2013-00-34%20Untitled0.ipynb%20-%20Colab.png)

## Setup dependencies

- Install the Hugging face `datasets` library
- Import base libraries for data handling like `pandas`, `numpy`, `os`, etc
- Load TensorFlow Framework
- Load `sklearn` for evalation

```python
#@title Setup dependencies
!pip install -q datasets

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
```

### Validate if a GPU is available

```python
#@title Check GPU
device_name = tf.test.gpu_device_name()
if device_name:
    print("✅ GPU available:", device_name)
else:
    print("⚠️ No GPU detected. Training will still work, but slower.")
```
