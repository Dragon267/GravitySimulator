import tensorflow as tf 
from extract import Extract
import numpy as np

extractor = Extract()

data = extractor.extract()

x_train = data[0]
y_train = data[1]

#print(x_train[0])
#print(y_train[1])

#x_train = np.array(x_train)
#y_train = np.array(y_train)


index = []
number = 57000

examples = []
answers = []

for i in range(len(y_train)):
    if y_train[i] == 0 and number >= 0:
        index.append(i)

        number -= 1
    if y_train[i] == 1:
        examples.append(x_train[i])
        answers.append(y_train[i])

print(index)

print(f"length of x_train => {len(x_train)}")
print(f"length of index => {len(index)}")

for i in range(len(index)):
    k = len(index) - i - 1
    x_train.pop(index[k])
    y_train.pop(index[k])

print(x_train[0])
print(y_train[0])

LAUNCH_EXAMPLES = 10

for i in range(LAUNCH_EXAMPLES):
    for g in range(len(examples)):
        x_train.append(examples[g])
        y_train.append(answers[g])


x_train = np.array(x_train)
y_train = np.array(y_train)

print(f"length of input => {len(x_train)}")


x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(2048, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(2048, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(4096, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(516, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(216, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation = tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

model.fit(x_train, y_train, epochs=5)

model.save('rocket_alpha_1.model')

