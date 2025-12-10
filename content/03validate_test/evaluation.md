---
title: "Evaluation of the trained Model"
linkTitle: "Evaluation of the trained Model"
weight: 2
---

## Evaluate the Model based on the Test Dataset

After training the Transformer spam classifier, the final step is to evaluate how well the model performs on completely unseen data. This is done using the test set, which acts like the model’s final exam. Because the model did not encounter this data during training or validation, the test results give a realistic estimate of how the model will behave in real-world scenarios.

The first part of the code uses model.evaluate(test_ds) to compute three important metrics on the test dataset: the loss, the accuracy, and the AUC. The loss tells us how far off the predictions are on average, while accuracy shows the proportion of messages the model classified correctly. AUC (Area Under the ROC Curve) is especially helpful in spam detection because it measures how well the model separates spam from ham across all possible classification thresholds. Together, these metrics provide a high-level view of the model’s performance.

However, high-level metrics alone are not enough to understand the behavior of a classifier. To dig deeper, we collect raw predictions for each individual email in the test set. The model outputs a probability between 0 and 1 representing how likely a message is to be spam. We convert these probabilities into binary class predictions using a threshold of 0.5: values above or equal to 0.5 are labeled as spam, and everything below as ham. This allows us to compare predictions directly against the true labels.

With the predicted and actual labels, we can compute more detailed evaluation measures. The classification report provides precision, recall, and F1-scores for both classes. These metrics are vital, especially when the dataset is imbalanced, because accuracy alone may hide important weaknesses. For example, a model might achieve high accuracy by simply predicting everything as ham, but precision and recall would reveal that it is failing completely at catching spam. The precision for spam tells us how many messages predicted as spam were actually spam, while recall tells us how many real spam messages the model successfully detected. The F1-score provides a balanced measure of both.

Lastly, the confusion matrix shows the exact number of correct and incorrect predictions for each class. This helps identify systematic errors: false positives (ham incorrectly marked as spam) and false negatives (spam the model failed to detect). In a spam classifier, the balance between these two types of errors is crucial. Too many false positives annoy users with messages being wrongly filtered out, while too many false negatives let harmful or unwanted spam slip through.

Overall, this evaluation code helps students understand not only how well the model performs overall but also how it performs, what types of mistakes it makes, and whether it is suitable for real-world use. It reinforces the idea that a proper machine learning evaluation always goes beyond accuracy and requires deeper diagnostic tools such as precision, recall, F1-score, and confusion matrices to truly understand a classifier’s strengths and weaknesses.

```python
#@title Evaluate on test set

test_loss, test_acc, test_auc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Collect predictions for detailed metrics
test_texts_list = list(test_texts)
test_labels_array = np.array(test_labels)

pred_probs = model.predict(tf.constant(test_texts_list)).ravel()
pred_labels = (pred_probs >= 0.5).astype("int32")

print("\nClassification report:")
print(classification_report(test_labels_array, pred_labels, target_names=["ham", "spam"]))

print("\nConfusion matrix:")
print(confusion_matrix(test_labels_array, pred_labels))
```

## Spam Classifier Interpretation Cheat Sheet

### Test Metrics (Loss, Accuracy, AUC)

#### Loss
- Measures how wrong the predictions are (lower is better)
- If test loss is much higher than train loss → overfitting
- If both losses stay high → underfitting or poor hyperparameters

#### Accuracy
- Percentage of correct predictions
- Can be misleading with imbalanced datasets

**Interpretation:**
- 90%+ is good
- Always compare accuracy with precision/recall

#### AUC (Area Under ROC Curve)
- Measures how well the model separates spam vs ham
- Threshold‑independent

**Interpretation:**
- 0.5 → random guessing
- 0.7–0.8 → decent
- 0.8–0.9 → strong
- 0.9+ → excellent


### Classification Report

#### Precision
“How many predicted spams were actually spam?”

- High precision → few false alarms
- Low precision → many ham→spam mistakes

**Interpretation:**
- Spam precision < 0.80 → too many false positives
- Ham precision < 0.95 → misclassification risk

#### Recall
“How many real spam messages did we catch?”

- High recall → strong spam detection
- Low recall → spam slipping through

**Interpretation:**
- Spam recall < 0.80 → missing too much spam

#### F1‑score
Balanced combination of precision and recall.

**Interpretation:**
- 0.70–0.80 → OK  
- 0.80–0.90 → strong  
- \>0.90 → excellent  

### Confusion Matrix

A typical matrix:

|               | Pred Ham | Pred Spam |
|---------------|----------|-----------|
| Actual Ham    | TN       | FP        |
| Actual Spam   | FN       | TP        |

#### TP — True Positives  
Correctly identified spam.

#### TN — True Negatives  
Correctly identified ham.

#### FP — False Positives  
Ham → predicted spam  
- Annoys users
- Reduce by increasing threshold or improving precision

#### FN — False Negatives  
Spam → predicted ham  
- Dangerous (spam gets through)
- Reduce by lowering threshold or improving recall

### Quick Decision Guide

#### High accuracy + low spam recall  
→ Model misses spam  
→ Lower threshold, tune hyperparameters

#### Low spam precision  
→ Flags too many ham messages  
→ Raise threshold

#### Loss decreases but validation metrics worsen  
→ Overfitting  
→ Add dropout, reduce epochs

#### AUC < 0.7  
→ Poor class separation  
→ Improve preprocessing or Transformer setup

### Ideal Metric Profile

- Accuracy ≥ 0.90  
- Spam precision ≥ 0.85  
- Spam recall ≥ 0.85  
- AUC ≥ 0.90  
- Balanced FP/FN rates  

