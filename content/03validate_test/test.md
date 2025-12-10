---
title: "Save/Load & Test the Model"
linkTitle: "Save/Load & Test the Model"
weight: 3
---

## Save the model to disk

In this task, you will save the model to disk for later use or to share with others.

```python
#@title Save model to disk

save_path = "enron_spam_transformer.keras"

model.save(
    save_path,
    include_optimizer=True,
)

print("Model saved to:", save_path)
```

## Load the model from disk

In this task, you will load the saved model from disk e.g for use in another script or application.

```python
#@title Load saved model

loaded_model = keras.models.load_model(
    save_path,
    custom_objects={"TransformerBlock": TransformerBlock},
)

loaded_model.summary()
```

## Use the loaded model to evaluate on custom input

This section demonstrates how to use the loaded model to classify new email texts as spam or ham by using your own input. Feel free to change the example inputs and add your own.

```python
#@title Test the trained model with custom input

def classify_email(text, model=loaded_model):
    text_tensor = tf.constant([text])
    prob = model.predict(text_tensor)[0][0]
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"Predicted label: {label}  (prob = {prob:.3f})")
    return prob, label

# Try some examples:
examples = [
    "Get cheap Viagra now!!! Limited offer, click here to buy.",
    "Hi team, the meeting is scheduled for tomorrow at 10 AM.",
    "You have won $1,000,000. Please send your bank details.",
    "Hi Marc, how are you doing? I hope this email finds you well.",
    "Hi, I have a super secret offer for you - 200% off of Clothes if you click this link",
]

for e in examples:
    print("=" * 80)
    classify_email(e)
```

