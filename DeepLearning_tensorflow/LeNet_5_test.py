# 作者 : 杨航
# 开发时间 : 2022/10/13 14:35
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
# 读取数据
train = pd.read_csv('F:/PythonRepository/Deep-Learing/SAT_MNIST/train.csv')
test = pd.read_csv('F:/PythonRepository/Deep-Learing/SAT_MNIST/test.csv')
# 数组转换
train = np.array(train)
test = np.array(test)

# 提取图片信息
train_images = train[:,1:]
test_images = test[:,1:]

# 提取标签信息
train_labels = train[:,:1]
test_labels = test[:,:1]
print(train_images.shape)
print(train_labels.shape)

# 将标签变为1维
# 将图片变为28*28
train_labels = train_labels.reshape(68161)
test_labels = test_labels.reshape(8529)

train_images = train_images.reshape(68161,28,28)
test_images = test_images.reshape(8529,28,28)

# 打印第101张图片
plt.imshow(train_images[100])

# 增加维度,用于卷积操作
train_images = train_images.reshape(68161,28,28,1)
test_images = test_images.reshape(8529,28,28,1)

# 归一化
train_images = train_images/255
test_images = test_images/255

# 独热编码
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
# 训练图片、训练标签作为输入。测试图片、测试标签作为每一轮的验证
history = model.fit(train_images,train_labels,epochs=50,validation_data=(test_images,test_labels))
# 轮次、正确率（训练集与测试集）
plt.plot(history.epoch,history.history.get('acc'))
plt.plot(history.epoch,history.history.get('val_acc'))
# 模型评估
model.evaluate(test_images,test_labels)
# 模型保存
model.save('sat.h5')

# 调用模型
new_model = tf.keras.models.load_model('sat.h5')
# 读取图片
img = cv2.imread('F:/PythonRepository/Deep-Learing/SAT_MNIST/68187.jpg',0)
img = cv2.resize(img,(28,28))
img = img.reshape(1,28,28,1)
img = img/255
# 预测
predict = new_model.predict(img)
print(predict)
label = ['car','harbor','helicopter','oil_gas_field','parking_lot','plane','runway_mark','ship','stadium','storage_tank']
result = label[np.argmax(predict)]
print(result)