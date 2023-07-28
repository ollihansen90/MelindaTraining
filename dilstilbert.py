# Load model directly
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/workingdir/.cache/huggingface/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workingdir/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/workingdir/.cache/huggingface/hub'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
torch.hub.set_dir("/workingdir/.cache")

satz = ":-("

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-german-cased")
model.pre_classifier = nn.Identity()
classifier = nn.Linear(768,3)
model.classifier = nn.Identity()

ttt = tokenizer([satz], return_tensors="pt").input_ids
print(ttt)
print(classifier(model(ttt).logits))
print(model)
print(sum([p.numel() for p in model.parameters()]))
print(tokenizer)
