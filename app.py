import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize le motif - mots divisés en tableau
    sentence_words = nltk.word_tokenize(sentence)
    # stem chaque mot - créer une forme courte pour le mot
    sentence_words = [
        lemmatizer.lemmatize(word.lower()) for word in sentence_words
        ]
    return sentence_words

# retour sac de mots : 
# 0 ou 1 pour chaque mot du sac qui existe dans la phrase

def bow(sentence, words, show_details=True):
    # marquer le modèle
    sentence_words = clean_up_sentence(sentence)
    # sac de mots - matrice de N mots, matrice de vocabulaire
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # attribuer 1 
                # si le mot courant est dans la position de vocabulaire
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filtrer les prévisions sous un seuil
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # trier par force de probabilité
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if len(ints) == 0:
        return "Je ne comprends pas. Pouvez-vous reformuler votre question, s'il vous plait?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/index', methods=['POST'])
def index():
    username = request.form.get('username')
    return render_template('index.html', username=username)

@app.route('/')
def connexion():
    return render_template('connexion.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
