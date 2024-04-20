# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:23:28 2024

@author: asus
"""

import tensorflow as tf
from tensorflow import keras

directory = 'C:/Users/asus/Desktop/Files/dars/bi/HCD/HCD/New/test'
test_data = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35"],
    color_mode="grayscale",
    batch_size=32,
    image_size=(8, 30),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    #data_format=None,
)
directory2 = 'C:/Users/asus/Desktop/Files/dars/bi/HCD/HCD/train/train_set'
train_data = keras.utils.image_dataset_from_directory(
    directory2,
    labels="inferred",
    label_mode="int",
    class_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35"],
    color_mode="grayscale",
    batch_size=32,
    image_size=(8, 30),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    #data_format=None,
)

model = keras.Sequential([
  keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
  keras.layers.MaxPooling2D((2,2)),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"),
  keras.layers.Dense(36, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Train the model
model.fit(train_data, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
predict = model.predict(test_data)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

#import cv2
#import numpy as np

# تابع برای پیش‌پردازش عکس
#def preprocess_image(image):
    # تغییر اندازه عکس به اندازه مورد نیاز شبکه عصبی
  #  image = cv2.resize(image, (30, 8))
    # تبدیل عکس به مقیاس خاکستری
 #   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # تغییر مقیاس پیکسل‌ها به بازه 0 تا 1
    #image = image / 255.0
    # تغییر ابعاد عکس به [1, ارتفاع, عرض, 1] برای ورودی شبکه عصبی
   # image = np.expand_dims(image, axis=0)
    #image = np.expand_dims(image, axis=-1)
    #return image


# بارگیری مدل آموزش دیده

# بارگیری و پیش‌پردازش عکس ورودی

#image = cv2.imread(image_path)
#preprocessed_image = preprocess_image(image)

# پیش‌بینی کلاس عکس
#prediction = model.predict(preprocessed_image)
#predicted_class = np.argmax(prediction)

# چاپ کلاس پیش‌بینی شده
class_names = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35"]
#predicted_class_name = class_names[predicted_class]
#print("Predicted class:", predicted_class_name)


from keras.preprocessing import image
import numpy as np

def predict_image(image_path):
    # بارگیری تصویر
    img = image.load_img(image_path, target_size=(8, 30), color_mode="grayscale")
    
    # تبدیل تصویر به آرایه NumPy
    x = image.img_to_array(img)
    
    # اضافه کردن بعد دسته‌ای
    x = np.expand_dims(x, axis=0)
    
    
    # پیش‌بینی کلاس
    prediction = model.predict(x)
    
    # دریافت شاخص کلاس با بیشترین احتمال
    predicted_class_index = np.argmax(prediction)
    
    # دریافت نام کلاس
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name


# دریافت مسیر تصویر از کاربر
image_path = "C:/Users/asus/Desktop/Files/dars/bi/project/99.jpg"

# پیش‌بینی کلاس تصویر
predicted_class_name = predict_image(image_path)

# نمایش کلاس پیش‌بینی‌شده
print("کلاس پیش‌بینی‌شده:", predicted_class_name)


    
