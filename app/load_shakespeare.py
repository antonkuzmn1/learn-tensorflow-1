import numpy as np
import requests
import tensorflow as tf

from app.custom_losses import loss

url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
response = requests.get(url)
with open('shakespeare.txt', 'wb') as f:
    f.write(response.content)

text = open('shakespeare.txt', 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# model = Sequential([
#     Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=seq_length),
#     LSTM(RNN_UNITS, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform'),
#     Dense(VOCAB_SIZE)
# ])

inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inputs)
x = tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform')(x)
outputs = tf.keras.layers.Dense(VOCAB_SIZE)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss=loss)
