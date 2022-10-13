# 作者 : 杨航
# 开发时间 : 2022/10/13 14:39
### 1.激活函数：LeNet-5使用sigmoid、AlexNet使用ReLU
####  a.模型收敛快，避免梯度消失
####  b.计算简单，运算速度快
### 2.随机失活Dropout
####   每次训练都随机让一定神经元停止参与运算，增加模型的泛化能力
####   稳定性和鲁棒性，避免过拟合。
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# GPU内存管理（tensorflow默认占用所有的显存，所以可以使用该代码）
# 以模型的大小来占用显存，比较合理
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True)
# 定义文件路径
train_dir = 'F:/PythonRepository/Deep-Learing/SAT/train'
test_dir = 'F:/PythonRepository/Deep-Learing/SAT/val'

# 网络要求图片的大小
im_size = 224
# 批处理大小（数值越大，越接近真实情况，占用的内存过大。过小，网络拟合有影响）
batch_size = 16

# 归一化(测试集图片进行水平翻转)
train_images = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True)
test_images = ImageDataGenerator(rescale=1 / 255)

# 批量打包（按路径载入图片，批处理大小，随机，尺寸，独热编码）
train_gen = train_images.flow_from_directory(directory=train_dir,
                                             batch_size=batch_size,
                                             shuffle=True,  # 随机载入（数据多样性）
                                             target_size=(im_size, im_size),
                                             class_mode='categorical')  # 多分类问题，所以将标签信息转为独热编码的形式)

# 批量打包（测试图片）
test_gen = test_images.flow_from_directory(directory=test_dir,
                                           batch_size=batch_size,
                                         shuffle=False,  # 随机载入（数据多样性）
                                           target_size=(im_size, im_size),
                                           class_mode='categorical')
# 类别查看
classes = train_gen.class_indices
print(classes)

# 搭建网络
model = tf.keras.Sequential()
# 非对称的填充，行列都加上3，使224变为227，满足要求
model.add(tf.keras.layers.ZeroPadding2D(((1,2),(1,2)),input_shape=(224,224,3)))
model.add(tf.keras.layers.Conv2D(filters=48,
                                kernel_size=(11,11),
                                strides=4,
                                activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
model.add(tf.keras.layers.Conv2D(filters=128,
                                kernel_size=(5,5),
                                padding='same',
                                activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
model.add(tf.keras.layers.Conv2D(filters=192,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=192,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2048,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2048,activation='relu'))
model.add(tf.keras.layers.Dense(5,activation='softmax'))

model.summary()
# 优化
# 学习率设置
# 默认adam优化器，会存在比较难拟合的情况，所以使用手动设置学习率

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
             loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(train_gen,epochs=15,validation_data=test_gen)
plt.plot(history.epoch,history.history.get('acc'))
plt.plot(history.epoch,history.history.get('val_acc'))
# 模型评估
model.evaluate(test_gen)
# 保存模型
model.save('sat2.h5')
# 调用模型
new_model = tf.keras.models.load_model('sat2.h5')
# 读取图片
img = cv2.imread('F:/PythonRepository/Deep-Learing/SAT/1.jpg',1)
plt.imshow(img)
# 图片预处理
img = cv2.resize(img,(224,224))
img = img.reshape(1,224,224,3)
img = img/255

# 预测
predict = new_model.predict(img)
print(predict)
label = ['airplane','bridge','palace','ship','stadium']
result = label[np.argmax(predict)]
print(result)