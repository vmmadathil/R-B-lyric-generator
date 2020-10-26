## this will be used to load the model and create predictions

import tensorflow as tf
import numpy as np
import json


#this function will load the model and checkpoints
def loadModel(vocab_size, embedding_dim, rnn_units, batch_size, checkpoint_dir):
    #loading the model
    tf.train.latest_checkpoint(checkpoint_dir)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])

    #model.summary()
    return model



#this function will make the prediction
def makePrediction(start_string, model):
    
    with open('lyrics.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    with open('additional_lyrics.txt', 'r', encoding='utf-8') as f:
        text2 = f.read()

    text = text + text2

    vocab = sorted(set(text))
    char2int = {c:i for i, c in enumerate(vocab)}
    int2char = np.array(vocab)

    print ('There are {} unique characters'.format(len(vocab)))
    num_generate = 1000

    input_eval = [char2int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(int2char[predicted_id])
    
    return(start_string + ''.join(text_generated))


def predict(start_string):

    checkpoint_dir = './checkpoints_2020'
    start_string = start_string

    model = loadModel(
        vocab_size = 72,
        embedding_dim = 256,
        rnn_units = 1024,
        batch_size = 1,
        checkpoint_dir = checkpoint_dir)


    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.summary()

    prediction = makePrediction(start_string, model)

    print(prediction)
    return prediction

if __name__ == "__main__":
    predict()
    