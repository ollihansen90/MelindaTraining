# Load model directly
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/workingdir/.cache/huggingface/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workingdir/.cache/huggingface/hub'
os.environ['HF_HOME'] = '/workingdir/.cache/huggingface/hub'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
torch.hub.set_dir("/workingdir/.cache")
from utils import Dataset
from tqdm.auto import tqdm
from math import cos

def main():
    satz = "Hi, mein Name ist Olli. Wie geht es dir heute?"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-german-cased")
    model.pre_classifier = nn.Identity()
    model.classifier = nn.Identity()
    classifier = nn.Linear(768,3)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    D = Dataset(model, tokenizer, "supernet")
    batch_size = 16
    dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    n_epochs = 1_000
    bar = tqdm(range(n_epochs))
    lr_start = 1e-3
    lr_end = 1e-5
    for epoch in bar:
        lr = lr_end + (lr_start - lr_end) * (1 + cos(epoch/n_epochs))/2
        classifier.train()
        l, n, a = 0, 0, 0
        for batch, label in dataloader:
            optimizer.zero_grad()
            output = classifier(batch)
            loss = criterion(output, label.long())
            acc = (output.argmax(1) == label).float().sum()
            a += acc.item()
            l += loss.item()
            n += len(label)
            
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            bar.set_description(f"[{epoch+1}/{n_epochs}] Loss: {l/n:.2f}, Acc: {a/n*100:.2f}%, lr: {lr:.2e}")

    eingabe = ""
    labeldict = {0: "good words", 1: "bad words", 2: "neutral words"}
    while eingabe!="exit":
        eingabe = input("Eingabe: ")
        ttt = tokenizer([eingabe], return_tensors="pt", padding=True, truncation=True)
        out = (classifier(model(ttt.input_ids, attention_mask=ttt.attention_mask).logits))
        print(out)
        print(labeldict[out.argmax().item()])

    #print(model)
    #print(sum([p.numel() for p in model.parameters()]))
    #print(tokenizer)
    return

if __name__ == '__main__':
    main()
