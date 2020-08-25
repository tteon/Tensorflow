import os
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True, # will return  tuple (img, label) otherwise dict
    with_info=True, # able to  get info about dataset

)

def normalize_img(image,label):
    '''normalize images'''
    return tf.cast(image, tf.float32) / 255.0 , label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE) # num_parallel_calls
ds_train = ds_train.cache() # cache
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE) #
ds_train = ds_train.prefetch(AUTOTUNE) # prefetch

model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        tf.keras.layers.Dense(10),
    ]
)

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

## Customize , source from ; tensorflow official site
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #print(logs.keys())
        if logs.get("accuracy") > 0.90: # many argument is there  ex) val_accuracy
            print("Accuracy over 90%, qutting training")
            self.model.stop_training = True


## default
save_callback = keras.callbacks.ModelCheckpoint(
    'checkpoint/',
    save_weights_only=True,
    monitor='accuracy',
    save_best_only=False,

)


model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

##
model.fit(
    ds_train,
    epochs=10,
    verbose=2,
    callbacks=[save_callback, lr_scheduler, CustomCallback()],

)