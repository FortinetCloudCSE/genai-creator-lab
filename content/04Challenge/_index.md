---
title: "Final Challenge"
linkTitle: "Challenge"
weight: 60
---

Welcome to the final challenge of this course! This challenge is designed to test your understanding of the concepts and skills you've learned throughout the modules.

Tweak and improve your model based on the feedback and insights you've gathered so far. Once you're satisfied with your model's performance, submit it for validation.

{{% notice tip %}}
While building your model, several variables have been left to be used for tweaking.
For Example:
- `max_tokens`
- `max_len`
- `batch_size`
- `ff_dim`
- `dropout_rate`
- `epochs`

{{% /notice %}}

## Tasks
- [ ] Review the course material and tasks
- [ ] Tweak and improve your model
- [ ] Upload your final model to get validated

## How to Submit Your Model

To submit your final model for validation, please use the following code snippet within your Colab notebook:

```python
#@title Submit Final Model for Validation
import requests, zipfile, os, json, time

PROVISIONER = "http://ml-workshop.ftntlab.tech/claim"

claim = requests.post(PROVISIONER, timeout=20).json()
submit_url = claim["submit_url"]
token = claim["submit_token"]

print("[*] Your submit URL:", submit_url)
print("[*] Your token:", token)

model.save("submission.keras")
!zip submission.zip submission.keras
print("[*] Model saved and zipped.")

time.sleep(5)

with open("submission.zip", "rb") as f:
    r = requests.post(
        submit_url,
        files={"file": f},
        headers={"X-Submit-Token": token},
        timeout=180
    )
print("[*] Your Result:\n")
if r.headers.get("content-type","").startswith("application/json"):
    print(json.dumps(r.json(), indent=2))
```

## Hints and Tips
{{% expand title="Hint 1" %}}
Have a look at the `epochs` parameter in the training step. Maybe slightly increasing it to something between 4 - 6 will help?
{{% /expand %}}

{{% expand title="Hint 2" %}}
Have a look at the `ff_dim` parameter in the Transformer block. Maybe increasing it will help to something between 32 - 128?
{{% /expand %}}

{{% expand title="Hint 3" %}}
Have a look at the `dropout_rate` parameter in the Transformer block. Maybe decreasing it will help to something between 0.1 - 0.3?
{{% /expand %}}

{{% expand title="Hint 4" %}}
Have a look at the `max_len` parameter in the text vectorization step. Maybe increasing it will help to something between 150- 250?
{{% /expand %}}

{{% expand title="Hint 4" %}}
Have a look at the `batch_size` parameter in the data preparation step. Maybe increasing it will help to something between 64 - 128?
{{% /expand %}}

{{% expand title="Hint 5" %}}
Have a look at the `max_tokens` parameter in the text vectorization step. Maybe increasing it will help to something between 15000 - 25000?
{{% /expand %}}

{{% expand title="Hint 6" %}}
Have a look at the optimizer `learning_rate` in the model compilation step. Maybe decreasing it will help to something between 0.0001 - 0.0005?
{{% /expand %}}