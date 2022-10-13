# 作者 : 杨航
# 开发时间 : 2022/10/13 14:32
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)= tf.keras.datasets.mnist.load_data()
# 增加维度，便于卷积操作
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)

# 归一化
train_images = train_images/255
test_images = test_images/255

# 标签的独热编码
train_labels = np.array(pd.get_dummies(train_labels))
test_labels = np.array(pd.get_dummies(test_labels))

# 搭建网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),input_shape=(28,28,1),padding='same',activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=120,kernel_size=(5,5),activation='sigmoid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84,activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()

# 模型优化（adam优化器，交叉熵损失函数，记录训练的正确率）
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

# 训练
# 训练图片、训练标签作为输入。测试图片、测试标签作为每一轮的验证
history = model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))

# 模型评估
model.evaluate(test_images,test_labels)
# 模型保存
model.save('mnist.h5')
# 调用模型
new_model = tf.keras.models.load_model('mnist.h5')

# 调用模型
new_model = tf.keras.models.load_model('mnist.h5')
# 读取图片(灰度图)
img = cv2.imread('F:/PythonRepository/Machine-Learing/picture/3.png',0)
# plt.imshow(img)
img = cv2.resize(img,(28,28)) # 将图片变为28*28的像素
img = img.reshape(1,28,28,1)  # 调整维度
img = img/255  # 归一化
predict = new_model.predict(img)
print(predict)
result = np.argmax(predict)
print(result)