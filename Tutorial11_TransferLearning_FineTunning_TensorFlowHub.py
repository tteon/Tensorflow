import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
from keras import applications
from keras.applications import densenet



# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0],True)

## Pretrained-Model
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = keras.models.load_model('complete_save_model/')

model.trainable = False # frezze
#print(model.summary())

## freeze
for layer in model.layers:
    assert layer.trainable == False
    layer.trainable = False


base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_output = layers.Dense(10)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=base_outputs)
print(new_model.summary())

new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


## Pretrained Keras Model
x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

model = keras.applications.InceptionV3(include_top=True)

base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(5)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
#print(model.summary())

new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

new_model.fit(x, y, epochs=15, verbose=2)

'''
## Pretrained Hub Model

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable = False # you are using fine-tunning

model = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5),
])

model.compile(
    optimizer=kears.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(x, y, epochs=15, verbose=2)