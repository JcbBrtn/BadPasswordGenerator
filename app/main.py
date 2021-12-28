from flask import render_template, Flask, request, redirect

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,GRU,Dense,LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy

import random
import numpy as np
import os

print(os.getcwd())

vocab = [' ', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char= np.array(vocab)

def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true,y_pred,from_logits=True)

def create_model(vocab_size,embed_dim,rnn_neurons,batch_size):
    model = Sequential()
    
    model.add(Embedding(vocab_size,embed_dim,batch_input_shape=[batch_size,None]))
    model.add(GRU(rnn_neurons,return_sequences=True, stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    
    model.compile(optimizer='adam',loss=sparse_cat_loss)
    
    return model

def generate_text(model,start_seed,gen_size=7,temp=1.0):
    num_generate = gen_size
    start_seed = ', ' + start_seed
    input_eval = [char_to_ind[s] for s in start_seed]
    
    input_eval = tf.expand_dims(input_eval,0)
    
    text_generated = []
    
    temperature = temp
    
    model.reset_states()
    
    for i in range(num_generate):
        preds = model(input_eval)
        preds = tf.squeeze(preds,0)
        preds = preds / temperature
        pred_id = tf.random.categorical(preds,num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([pred_id],0)
        text_generated.append(ind_to_char[pred_id])
        
    s = start_seed + "".join(text_generated)
    r = s.split(',')
    return r[1]

app = Flask(__name__)

tstmodel = create_model(len(vocab),embed_dim=64,rnn_neurons=1026,batch_size=1)
tstmodel.load_weights('./PasswordGen.h5')
tstmodel.build(tf.TensorShape([1,None]))

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'GET':
            pw=None
        elif request.method=='POST':
            pw = generate_text(tstmodel, "", gen_size=13, temp=0.8)
    except:
        pw=None
    finally:
        return render_template('main.html', pw=pw)


if __name__=='__main__':
    app.run()