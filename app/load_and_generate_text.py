import tensorflow as tf
from app import load_shakespeare as ss
from app.custom_losses import loss

model = tf.keras.models.load_model('shakespeare_text_generation_model.keras', custom_objects={'loss': loss})


def generate_text(rnn_model, start_string):
    num_generate = 100
    input_eval = [ss.char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    for i in range(num_generate):
        predictions = rnn_model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(ss.idx2char[predicted_id])

    return ''.join(text_generated)


# print('generated text:\n', generate_text(model, start_string=input('prompt: ')))
