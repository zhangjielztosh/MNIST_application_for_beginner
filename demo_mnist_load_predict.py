import numpy as np
np.bool = np.bool_
import keras
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image


# load model
model = keras.models.load_model('./my_mnist_model.keras')

# load test img
img_test = Image.open('./mytest.bmp')

# resize to 28 * 28
img_resize = img_test.resize((28,28))
# plt.imshow(img_resize)
# plt.show()

# convert to gray-scale
img_test_L = img_resize.convert('L')
# plt.imshow(img_test_L)
# plt.show()

# transfer img to array
img_np = np.array(img_test_L)
# print("img_np shape:", img_np.shape)
# print("img_np", img_np)

# convert with 255 black to white
img_np_revert = 255 - img_np
# print("img_np revert", img_np_revert)

# convert to float and expand dims to fit the input data format (1, 28, 28, 1)
img_self_input = img_np_revert.astype("float32") / 255
# print("img_self_input", img_self_input)
img_self_input = np.expand_dims(img_self_input, -1)
# print("img_self_input after dims", img_self_input)
# print("img_self_input shape", img_self_input.shape)
img_self_input_final = np.expand_dims(img_self_input, axis=0)
# print("img_self_input_final shape", img_self_input_final.shape)
# print("img_self_input_final", img_self_input_final)


# print("x_train[1:2] shape:", x_train[1:2].shape)
# print("x_train[1:2]:", x_train[1:2])
# ret = model.predict(x_train[0:1], batch_size=1) 

ret = model.predict(img_self_input_final, batch_size=1) 
print("predict ret:", ret)
retlist = ret.tolist()[0]
print("max:", max(retlist))
print("index:", retlist.index(max(retlist)))
print("identify as:", retlist.index(max(retlist)))





