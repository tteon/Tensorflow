import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# If you use GPU this might save you some errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# this movie was terrible -> 0
# this movie was really good -> 1

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True, # when training
    with_info=True,
)
'''
checking the dataset 
print(ds_info)

for text, label in ds_train:
    print(text)
    import sys
    sys.exit()
'''

# i loved this movie -> [TOKENIZATION] [ 'i', 'loved', 'this', .]
tokenizer = tfds.features.text.Tokenizer()

def build_vocabulary():
    vocabulary = set()
    for text, _ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))
    return vocabulary

vocabulary = build_vocabulary()

encoder = tfds.features.text.TokenTextEncoder(
    vocabulary, oov_token = "<UNK>", lowercase=True, tokenizer=tokenizer
)

def my_encoding(text_tensor, label):
    return encoder.encode(text_tensor.numpy()), label


### i have no idea about below codes ..
# we need to do a funciton
# part of the graph
# specifiy go through some python function
def encode_map(text, label):
    encoded_text, label = tf.py_function(
        my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64) # part of graph
    )

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ())) # specifying which of the of the shapes that are padded
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(encode_map)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential([
    layers.Masking(mask_value=0), # dont perform any computation
    layers.Embedding(input_dim=len(vocabulary)+2, output_dim=32),
    # BATCH_SIZE x 1000 -> BATCH_SIZE x 1000 x 32
    layers.GlobalAveragePooling1D(),
    # BATCH_SIZE x 32
    layers.Dense(64, activation='relu'),
    layers.Dense(1), # less than 0 negative, greater or eqaul 0 positive
])

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4, clipnorm=1), # we don't get exploding gradient problems
    metrics=['accuracy'],
)

model.fit(ds_train, epochs=10, verbose=2)
model.evaluate(ds_test)
