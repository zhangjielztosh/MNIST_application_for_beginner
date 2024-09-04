import numpy as np
np.bool = np.bool_ # solve error "module 'numpy' has no attribute 'bool'"
import keras
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# show one training sample, e.g.(x_train[0], y_train[0])
# img_tmp = (28,28)
# img_tmp = x_train[0]
# plt.imshow(img_tmp)
# plt.show()
# print(y_train[0])

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 128
epochs = 5

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save("./my_mnist_model.keras")

# predict one sample, e.g. x_train[0:1]
# print("x_train[1:2] shape:", x_train[0:1].shape)
# print("x_train[1:2]:", x_train[0:1])
# ret = model.predict(x_train[0:1], batch_size=1)
# print("predict ret:", ret)
