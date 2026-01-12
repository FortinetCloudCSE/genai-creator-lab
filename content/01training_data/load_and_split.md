---
title: "Prepare training, validation & test data"
linkTitle: "Prepare training, validation & test data"
weight: 2
---

## Load the Dataset
In this lab you will use the [Enron Spam Dataset](https://huggingface.co/datasets/SetFit/enron_spam) to train the model. This Dataset contains about 32.7k rows of different Spam and non-Spam emails.
The dataset has been already split into a training and a test dataset. The training dataset part (usually 70% of the dataset) will be used to train the model. The test dataset part (the other 30 %) will be used by the model to validate the learned information.

To load the dataset we will use the `load_dataset` function. In addition this code will load and store each dataset in a variable using pandas for the inspection to provide you an overview on how much data is loaded and how the data is formatted.

```python
#@title Load Enron-Spam dataset

dataset = load_dataset("SetFit/enron_spam")

print(dataset)
print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

# Convert to pandas for easier inspection / plotting
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

train_df.head()
```

To further evaluate the data, let's look at the distribution of the "good" (ham) and "bad" (spam) data. For this, use the variables from the previous step and create diagram of the data distribution.

For this, create a diagram plot using the type `bar` and print this on the console. In addition, let's have a look on some examples for spam/ham.
```python
#@title Evaluate the label distribution (training data)

print(train_df["label"].value_counts())
print(train_df["label"].value_counts(normalize=True))

train_df["label_text"].value_counts().plot(kind="bar")
plt.title("Label distribution (train)")
plt.xticks(rotation=0)
plt.show()

print("Example SPAM email:")
print(train_df[train_df["label_text"] == "spam"]["text"].iloc[0][:500])

print("\nExample HAM email:")
print(train_df[train_df["label_text"] == "ham"]["text"].iloc[0][:500])
```

## Splitting the Training Dataset

Most HuggingFace datasets provide only:
- a train set
- a test set

However, machine-learning workflows require a validation set as well.  This code manually creates an 80/20 split of the training data.
to achive this, the parameter `test_size=0.2`is used. This Sets the proportion of the validation set to 20%.
The Remaining 80% becomes the training data which are required for model tuning and avoiding overfitting.

The `seed=42`parameter sets a fixed random seed so the split is reproducible. Without a seed, each run would shuffle the data differently.

```python
#@title Create train / validation / test splits

split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds_hf = split["train"]
val_ds_hf = split["test"]
test_ds_hf = dataset["test"]

print("Train:", len(train_ds_hf))
print("Val:", len(val_ds_hf))
print("Test:", len(test_ds_hf))

# FORCE into plain Python lists of strings
train_texts = list(train_ds_hf["text"])
val_texts   = list(val_ds_hf["text"])
test_texts  = list(test_ds_hf["text"])

train_labels = np.array(train_ds_hf["label"], dtype="int32")
val_labels   = np.array(val_ds_hf["label"], dtype="int32")
test_labels  = np.array(test_ds_hf["label"], dtype="int32")
```