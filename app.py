from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
# 	model= load_model(spam_classifier.h5)
# 	tokenizer = pickle.load("tokenizer.pkl")
# 	if request.method == 'POST':
# 		message = request.form['message']
# 		data = [message]
# 		X = tokenizer.texts_to_sequences(data)
# 		X = pad_sequences(X, maxlen=100)
# 		my_prediction = model.predict_classes(X).flatten()[0]
# 	return render_template('result.html',prediction = my_prediction)



# if __name__ == '__main__':
# 	app.run(debug=True)

@app.route('/translate', methods=['POST', 'GET'])
def get_translation():
    if request.method == 'POST':
 
        result = request.form
        # Get the English sentence from the Input site
        engSentence = str(result['input_text'])
        # Converting the text into the required format for prediction
        # Step 1 : Converting to an array
        engAr = [engSentence]
        # Clean the input sentence
        cleanText = cleanInput(engAr)
        # Step 2 : Converting to sequences and padding them
        # Encode the inputsentence as sequence of integers
        seq1 = encode_sequences(Eng_tokenizer, int(Eng_stdlen), cleanText)
        # Step 3 : Get the translation
        translation = generatePredictions(model,Amh_tokenizer,seq1)
        # prediction = model.predict(seq1,verbose=0)[0]
 
        return render_template('result.html', trans=translation)

if __name__ == '__main__':
	app.run(debug=True)