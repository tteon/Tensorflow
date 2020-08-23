import os
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import layers
import tensorflow_datasets as tfds

# If you use GPU this might save you some errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=["train", "test"],
    shuffle_files = True,
    #as_supervised = False, # when you show example
    as_supervised = True, # (img, label)
    with_info = True,
)

# fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4) # visualization sample
# print(ds_info)
## DATA PREPROCESSING
def normalize_img(image, label):
    # normalize images
    return tf.cast(image, tf.float32)/255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache() # keep track of some of them in memory so that it's going to be faster for the next time
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE) # instantly call and utilize

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

model = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

model.fit(ds_train, epochs=5, verbose=2)
model.evaluate(ds_test)

