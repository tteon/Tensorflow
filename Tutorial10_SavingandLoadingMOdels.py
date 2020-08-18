import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# 1. How to save and load model weights
# 2. Save and load entire model (Serializing model)
## - Save weights
## - Model architecture
## - Training Configuration (model.compile())
## - Optimizer and states
# sequential API
model1 = keras.Sequential(
    [
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)

# functional API
inputs = keras.Input(784)
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)

    # subclass modeling for load model (
    def

model3 = MyModel()

# Specify model build from the three API's
model = keras.models.load_model('complete_save_model/')
#model = model1
# model weights incompatible -> before using weights and think about matching these API property
model.load_weights('saved_model/')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test,  y_test, batch_size=32, verbose=2)
model.save('complete_save_model/')

#model.save_weights('saved_model/')
# Plz check tf offical documentation because of exchanging default weights format than before