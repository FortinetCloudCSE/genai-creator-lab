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
import requests, zipfile, os, json

PROVISIONER = "http://ml-workshop.ftntlab.tech/claim"

claim = requests.post(PROVISIONER, timeout=20).json()
submit_url = claim["submit_url"]
token = claim["submit_token"]

print("Your submit URL:", submit_url)
print("Your token:", token)

SUBMISSION_DIR = "submission_model"
model.export(SUBMISSION_DIR)

zip_path = "submission.zip"
!zip -r {zip_path} {SUBMISSION_DIR}

with open(zip_path, "rb") as f:
    r = requests.post(
        submit_url,
        files={"file": f},
        headers={"X-Submit-Token": token},
        timeout=120
    )

print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))
```