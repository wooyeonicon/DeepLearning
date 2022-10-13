# 作者 : 杨航
# 开发时间 : 2022/10/13 14:42
### 1.堆叠两个3*3卷积核替代一个5*5卷积核
###    堆叠两个3*3卷积核替代一个7*7卷积核
###     相同感受野，训练参数量减少
### 2.增加网络深度，提升性能（小数据集可以使用迁移学习）
### 3.计算资源问题
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'F:/PythonRepository/Deep-Learing/SAT/train'
test_dir = 'F:/PythonRepository/Deep-Learing/SAT/val'
im_size = 224
batch_size = 32
# 归一化
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

# 建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 64,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 64,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 128,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 128,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 256,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 256,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 64,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(3,3),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                filters = 512,
                                kernel_size=(1,1),
                                padding='same',
                                activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.Dense(4096,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.Dense(4096,activation='relu'))
model.add(tf.keras.layers.Dense(5,activation='softmax'))
model.summary()

# 优化
# 学习率设置
# 默认adam优化器，会存在比较难拟合的情况，所以使用手动设置学习率

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
             loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(train_gen,epochs=10,validation_data=test_gen)

plt.plot(history.epoch,history.history.get('acc'))
plt.plot(history.epoch,history.history.get('val_acc'))

model.evaluate(test_gen)
model.save('sat3.h5')

# 读取图片
img = cv2.imread('F:/PythonRepository/Deep-Learing/SAT/1.jpg',1)
plt.imshow(img)
model1 = tf.keras.models.load_model('sat3.h5')
# 图片预处理
img = cv2.resize(img,(224,224))
img = img.reshape(1,224,224,3)
img = img/255
predict = model1.predict(img)
print(predict)
label = ['airplane','bridge','palace','ship','stadium']
result = label[np.argmax(predict)]
print(result)