---
title: "Optimize the Dataset"
linkTitle: "Optimize the Dataset"
weight: 4
---

## Optimizing the Dataset for TensorFlow

TensorFlow’s tf.data API provides a fast, scalable, and memory-efficient way to load data during training.
Instead of manually feeding NumPy arrays to the model, we wrap them into Dataset pipelines, which handle:
- batching
- shuffling
- prefetching
- efficient loading on CPU/GPU

The code below converts your processed text + labels into TensorFlow-ready datasets for training, validation, and testing.

```python
#@title Build TensorFlow Dataset

batch_size = 256  # can be tuned

def make_dataset(texts, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_texts, train_labels, training=True)
val_ds   = make_dataset(val_texts, val_labels, training=False)
test_ds  = make_dataset(test_texts, test_labels, training=False)

train_ds
```
You can adjust the `batch_size` variable if you like. A batch is a group of samples processed together in one forward/backward pass.
After your changes, monitor the model performance, the training speed and the memory usage of the model if you like.

Now that the data has been formatted and prepared for the machine learning model, proceed to the next chapter where you will build the actual model.