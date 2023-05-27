#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    if int_features[2] == 1:
        int_features.insert(0, 3)
    elif int_features[2] == 0:
        int_features.insert(1, 3)
    elif int_features[2] == 2:
        int_features[2] = 0
        int_features.insert(0, 3)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The estimated salary is $ {:0.2f}'.format(output))


# In[5]:


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# In[6]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




