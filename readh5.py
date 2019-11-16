import h5py
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

# filename = "final_training_model.h5"

loaded_model = tf.keras.models.load_model('final_training_model.h5')

# loaded_model.summary()
# print(loaded_model.layers[0])

IMG_SIZE = 30

loaded_model.layers[0].input_shape # (None, 26, 26, 32)

# batch_holder = np.zeros((4, IMG_SIZE, IMG_SIZE, 3))
# print(batch_holder[0, :])
# img_dir='test_set/'
# for i, img in enumerate(os.listdir(img_dir)):
# 	img = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))
# 	batch_holder[i, :] = img

# result=loaded_model.predict_classes(batch_holder)
 
# fig = plt.figure(figsize=(20, 20))

# print(result)
 
# for i, img in enumerate(batch_holder):
# 	fig.add_subplot(1,4, i+1)
# 	print(result[i])
# 	plt.title(result[i])
# 	plt.imshow(img/256.)
  
# plt.show()

image_path="./test_set/image0 (copy).png"
img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
plt.imshow(img)
img = np.expand_dims(img, axis=0)
result=loaded_model.predict_classes(img)
plt.title(get_label_name(result[0][0]))
plt.show()

# from keras.models import load_model
# import cv2
# import numpy as np

# model = load_model('final_training_model.h5')

# img = cv2.imread('./test_set/image0 (copy).png')
# # img = cv2.resize(img,(52,5))
# img = np.reshape(img,[1,54,65,3])

# classes = model.predict_classes(img)

# print(classes)
