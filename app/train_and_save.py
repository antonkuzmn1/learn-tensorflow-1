import tensorflow as tf
from app import load_shakespeare as ss


def train_and_save(epochs: int):
    history = ss.model.fit(ss.dataset, epochs=epochs)

    tf.keras.models.save_model(ss.model, 'shakespeare_text_generation_model.keras')
