# main.py
# Trainiert nacheinander alle Gruppenchatbots
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import trange
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

import nltk

# STEMMER (hier am besten Cistem statt Lancaster)
STEMMER = nltk.stem.lancaster.LancasterStemmer()

import matplotlib.pyplot as plt
import os
import json
from utils import Classifier # ACHTUNG: utils.py wird hier benötigt!

# Beziehe alle Chatbotnamen
gruppenliste = [name.split(".")[0] for name in os.listdir("data") if "json" in name]

# Bereits trainierte werden nicht nochmal trainiert (etwas unschön umgesetzt, bei einer Hand voll Bots geht das aber)
#trained = ["Supernet", "Salzwerk", "Gruppe_1", "melinda", "Gruppe", "LuSo", "MarzInator"]
trained = ['Gruppe', 'LuSo', 'MarzInator', 'Melinda', 'PommesBot', 'Salzwerk', 'Supernet', "Gruppe_1", "Discovery", "Dornröschen"]
trained = []
#gruppenliste = ["Supernet"]


# Durchlaufe alle Gruppennamen und trainiere jeweils den Chatbot
for gruppe in gruppenliste:
    # Überspringe bereits trainierte Bots (unschön umgesetzt, s.o.)
    if gruppe in trained:
        continue

    print(20*"-", "Training", gruppe, 20*"-")
    # Lade Trainingsdaten in den Arbeitsspeicher
    with open("data/"+gruppe+".json", encoding="utf-8") as file:
        intentsdata = json.load(file)
        #print(intentsdata)
    # Lade Stopwords, falls sie existieren (die werden ignoriert)
    with open("data/stopwords/stopwords.txt", "r", encoding="utf-8") as file:
        # Allgemeine Stopwords
        stopwords = [w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"]
    if gruppe+"_stop.txt" in os.listdir("data/stopwords"):
        with open("data/stopwords/"+gruppe+"_stop.txt", "r", encoding="utf-8") as file:
            # Gruppeneigene Stopwords
            stopwords.extend([w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"])
            stopwords = list(set(stopwords))
    stopwords = list(set([w.lower() for w in stopwords]))

    # Generiere Trainingsdaten
    words = [] # Liste bekannter Wörter in Tokenform
    labels = [] # zugehöriges Label (good, bad, neutral)
    docs_x = [] # Liste aller Sätze im Datensatz (anders als words, da words nur einzelne Wörter enthält)
    docs_y = [] # Label zu docs_x
    for intent in intentsdata["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    print(len(words))
    words = [w.lower() for w in words if not w.lower() in stopwords] # Stopwords aus words entfernen
    words = [STEMMER.stem(w.lower()) for w in words if w != "?"] # Token in Wortstämme umwandeln
    words = sorted(list(set(words))) # list(set(...)) löscht doppelte Einträge
    print(len(words))    

    with open("data/words/"+gruppe+"_words.txt", "w", encoding="utf-8") as file:
        # Speichere bekannte Wörter ab
        file.writelines([w+"\n" for w in words])
    labels = sorted(labels)
    print(labels)
    print(len(words))

    # Trainingsdaten vorbereiten
    training = [] 
    output = []
    out_empty = [0 for _ in range(len(labels))] # Lange Liste mit Nullen

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [STEMMER.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    # Supernet wird auf den Trainingsdaten ALLER Gruppen trainiert (alt)
    if gruppe=="Supernet":
        N = len(training)
        n_klasse = 400

        for n in [3,4,5,6]:
            for i in range(3*n_klasse):
                output_row = [0 for _ in range(len(labels))]
                output_row[i//n_klasse] = 1
                output.append(output_row)
                bag = [0 for _ in range(len(training[0]))]
                idx = 0
                idx_list = []
                j_list = []
                while sum(bag)<n and idx not in idx_list:
                    idx = torch.randint(N, (1,)).item()
                    if output[idx].index(1)==i//n_klasse:
                        for j, entry in enumerate(training[idx]):
                            if entry==1:
                                j_list.append(j)
                                bag[j] = 1
                with open("j_list.txt", "a") as file:
                    file.write(str(j_list)+"\n")
                training.append(bag)

    training = torch.tensor(training).float().to(device)
    output = torch.tensor(output).float().to(device)

    training = torch.cat((training, torch.zeros(int(len(training)/3), training.shape[1]).to(device)))
    output = torch.cat((output, torch.zeros(int(len(output)/3), 3).to(device)))
    print(training.shape, output.shape)

    # Initialisiere MLP (siehe utils.py) und Optimizer (hier Adam)
    model = Classifier([len(words), int(len(words)/2), len(labels)]).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    loss_func = F.cross_entropy
    n_epochs = 10000 # Meistens VIEL zu viele Epochen, kann früher abgebrochen werden, wenn sich nicht mehr viel ändert (oft weniger als 400)
    lossliste = torch.zeros(n_epochs)

    for epoch in trange(n_epochs):
        # Standard-Trainingsloop
        optimizer.zero_grad()
        out = model(training)
        loss = loss_func(out, output)
        loss.backward()
        optimizer.step()
        lossliste[epoch] = loss.item()
        if epoch%int(n_epochs/10)==0:
            print(epoch, loss.item())
    model.eval()
    
    # Plotte Auswertungen
    plt.figure()
    plt.plot(lossliste.cpu().numpy())
    plt.title("Losses "+gruppe)
    plt.grid()
    plt.savefig("outputs/plots/"+gruppe+"_loss.png")
    plt.figure()
    plt.plot(torch.log(lossliste).cpu().numpy())
    plt.title("Loglosses "+gruppe)
    plt.grid()
    plt.savefig("outputs/plots/"+gruppe+"_logloss.png")

    # Speichere Model und state_dict
    torch.save(model, "outputs/networks/"+gruppe+"_model.pt")
    torch.save(model.state_dict(), "outputs/networks/state_dicts/"+gruppe+".pt")

    


