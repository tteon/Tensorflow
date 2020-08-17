import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore information message from tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#print(y_train.shape)
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0 # -1 means keep whatever the value is on that dimension so in this case 60,000
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Sequential API ( Very convenient, not very flexible ) for example, you it only allows you have to one input mapped to one output
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'), # fully connected layer
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
# print(model.summary()) # debugging tool
model.add(layers.Dense(256, activation='relu' , name='my_layer'))
model.add(layers.Dense(10))

# checking layer's feature after tunning specific layer's name
'''
model = keras.Model(inputs=model.inputs,
                    outputs=[model.get_layer('my_layer').output])
'''
# checking all of these layers
model = keras.Model(inputs=model.inputs,
                    outputs=[layer.output for layer in model.layers])

features = model.predict(x_train)

for feature in features:
    print(feature.shape)

# checking the output of specific layer
'''
model = keras.Model(inputs=model.inputs,
                    outputs=[model.layers[-2].output])

feature = model.predict(x_train)
print(feature.shape)
'''
# this technique is checking above lines where happens its error , In brief just running above lines not below things
import sys
sys.exit()


'''
print(model.summary())
import sys
sys.exit()
'''

# Functional API ( A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name = 'first_layer')(inputs)
x = layers.Dense(256, activation='relu', name = 'second_layer')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

# sepecifies the network configurations, loss function , optimizer ...
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),# we going to use speicfy loss function, from_logits=True means largest value equals true
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)




