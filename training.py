import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize chaque mot
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #ajouter des documents dans le corpus
        documents.append((w, intent['tag']))

        # ajouter à notre liste de classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize et abaisser chaque mot et supprimer les doublons
words = [
            lemmatizer.lemmatize(w.lower()) 
            for w in words if w not in ignore_words
        ]
words = sorted(list(set(words)))
# trier les classes
classes = sorted(list(set(classes)))
# documents = combinaison de motifs et d’intentions
print (len(documents), "documents")
# classes = intentions
print (len(classes), "classes", classes)
# words = tous les mots, vocabulaire
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))

# créer nos données d'entraînement
training = []
# créer un tableau vide pour notre sortie
output_empty = [0] * len(classes)

# ensemble d'entraînement, sac de mots pour chaque phrase
for doc in documents:
    # initialiser notre sac de mots
    bag = []
    # liste des mots tokenized pour le modèle
    pattern_words = doc[0]
    # lemmatiser chaque mot 
    # - créer le mot de base, en essayant de représenter les mots associés
    pattern_words = [
        lemmatizer.lemmatize(word.lower()) for word in pattern_words
    ]
    # créer notre tableau de mots avec 1, 
    # si la correspondance de mots se trouve dans le modèle actuel
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # la sortie est un '0' pour chaque balise 
    # et '1' pour la balise courante (pour chaque motif)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# mélanger nos caractéristiques et se transformer en np.array
random.shuffle(training)
training = np.array(training)
# créer des listes de trains et d’essais. X - motifs, Y - intentions
train_x = list(training[:,0])
train_y = list(training[:,1])
print("donnees d'entraînement crees")


# Créer un modèle - 3 couches. Première couche 128 neurones, 
# deuxième couche 64 neurones et 3ème couche de sortie contient le nombre de neurones.
# égale au nombre d’intentions pour prédire l’intention de sortie avec softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiler le modèle. La descente en gradient stochastique 
# avec gradient accéléré Nesterov donne de bons résultats pour ce modèle
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# montage et sauvegarde du modèle 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("modele cree")