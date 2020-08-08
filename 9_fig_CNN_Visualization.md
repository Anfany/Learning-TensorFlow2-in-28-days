### 第9天：TensorFlow2构建数据管道—图像格式

+ **涉及到的知识点**
  + 卷积神经网络可视化
    + 中间层输出可视化
    + 卷积核可视化
    + 类激活图
  + 训练好的模型读取

### <font color=#0099ff size=4 face="微软雅黑">实例：TensorFlow花卉</font>

本数据集一共有3670张图片，图片大小不一，共5类：daisy（雏菊）, dandelion（蒲公英）, roses（玫瑰）, sunflowers（向日葵）, tulips（郁金香）。每一类为一个文件夹。

### 1，数据获取


```python
import pathlib  # 类似于os，但是比os好用
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras import models
plt.rcParams['font.family'] = 'Arial Unicode MS' 
plt.rcParams['axes.unicode_minus']=False
from math import ceil
from sklearn.model_selection import train_test_split
print('tensorflow版本:{}'.format(tf.__version__))
```

    tensorflow版本:2.1.0
    


```python
# 下载数据集
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228818944/228813984 [==============================] - 918s 4us/step
    C:\Users\GWT9\.keras\datasets\flower_photos
    


```python
data_root =pathlib.Path(r'C:\Users\GWT9\.keras\datasets\flower_photos')
```

查看每个类中包含图片的个数，并随机显示该类的1张图片。


```python
# 存储所有的图片的路径
ALL_FIG_PATH = []
# 存储所有的标签
ALL_LABEL = []
# 存储类别、编号的字典
start_calss_sign =0
class_dict = {}  # 类别、编号字典
class_list = []  # 类别列表
for item in data_root.iterdir():
    if pathlib.Path.is_dir(item):
        # 图片文件列表
        fig_list = list(item.iterdir())
        ALL_FIG_PATH += [str(fip) for fip in fig_list] # 路径格式变为字符串
        ALL_LABEL += [start_calss_sign] * len(fig_list)
        # 文件夹名称
        dir_name = item.name
        class_list.append(dir_name)
        class_dict[start_calss_sign] = dir_name
        start_calss_sign += 1
        # 图片个数
        print('{:10s}：{}张'.format(dir_name, len(fig_list)))
        # 显示图片
        for fig in np.random.choice(fig_list, 1):
            display.display(display.Image(fig))
```

    daisy     ：633张
    


![jpeg](output_8_1.jpg)


    dandelion ：898张
    


![jpeg](output_8_3.jpg)


    roses     ：641张
    


![jpeg](output_8_5.jpg)


    sunflowers：699张
    


![jpeg](output_8_7.jpg)


    tulips    ：799张
    


![jpeg](output_8_9.jpg)


### 2、图像数据转换

所有图片的路径ALL_FIG_PATH，对应的标签ALL_LABEL，标签和真实类别对应的字典class_dict。


```python
# 将数据变为同样大小的，并且归一化
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])  # 统一大小
    image /= 255.0  # 归一化
    return image
```


```python
# 根据路径读取图片数据
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
```

### 3、构建数据通道

首先分割数据集，将数据集合按照比例分为训练、验证以及测试数据集。


```python
# 按比例分割数据集合：训练、非训练
X_train, X_no_train, Y_train, Y_no_train = train_test_split(ALL_FIG_PATH, ALL_LABEL, test_size=0.33, stratify=ALL_LABEL)  # 保持比例
# 将非训练分割为验证和测试
X_val, X_test, Y_val, Y_test = train_test_split(X_no_train, Y_no_train, test_size=0.33, stratify=Y_no_train)
# 训练图片的数量
fig_count = len(X_train)
```

将图片路径和图片的标签合并在一起，形成数据集形式


```python
def get_dataset(figpath, figlabel):
    #  利用from_tensor_slices进行进行特征切片
    ds = tf.data.Dataset.from_tensor_slices((figpath, figlabel))
    # 路径变为图片
    def load_and_preprocess_from_path_label(path, label):
        return load_and_preprocess_image(path), label
    
    return ds.map(load_and_preprocess_from_path_label)
```


```python
# 训练数据集
TRAIN_DATA = get_dataset(X_train, Y_train)
# 验证数据集
VAL_DATA = get_dataset(X_val, Y_val)
# 测试数据集
TEST_DATA = get_dataset(X_test, Y_test)
```

将训练数据进行随机转换，并将数据存储在缓存文件中，提升模型训练的效率。


```python
BATCH_SIZE  = 64  # 批次训练的样本数
train_ds = TRAIN_DATA.cache(filename='./cache.tf-data')  # 定义缓存文件，提升运行效率
train_ds = train_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=fig_count))  # 随机打乱的缓冲定义为全部图片的个数，有助于打乱数据
train_ds = train_ds.batch(BATCH_SIZE).prefetch(1)  # 训练数据
val_ds = VAL_DATA.batch(BATCH_SIZE) # 验证数据
```

    WARNING:tensorflow:From <ipython-input-9-f3e34f9a1131>:3: shuffle_and_repeat (from tensorflow.python.data.experimental.ops.shuffle_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by `tf.data.Dataset.repeat(count)`. Static tf.data optimizations will take care of using the fused implementation.
    

### 4、构建CNN模型

+ ### 4.1 模型建立

利用Sequential按层顺序创建CNN模型


```python
### 建立模型
def build_cnn(name='CNN_V'):
    # 输入层
    in_put = tf.keras.Input(shape=(192, 192, 3), name='INPUT')
    
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], activation='relu', name='CONV_1')(in_put)
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], activation='relu', name='CONV_2')(x)
    # 最大池化层
    x = tf.keras.layers.MaxPooling2D(pool_size=[2,2], name='MAXPOOL_1')(x)

    # 卷积层
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], activation='relu', padding='same', name='CONV_3')(x)
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], activation='relu', padding='same', name='CONV_4')(x)
    # 最大池化层
    x = tf.keras.layers.MaxPooling2D(pool_size=[2,2], name='MAXPOOL_2')(x)
    
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], activation='relu', padding='same', name='CONV_5')(x)
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], activation='relu', padding='same', name='CONV_6')(x)
    # 卷积层
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=[1,1], activation='relu', padding='same', name='CONV_7')(x)

    # 全局平均池化层，和平铺层相比，有助于减少参数
    x = tf.keras.layers.GlobalAveragePooling2D(name='GAP_1')(x)
    
    # 输出层
    out_put = tf.keras.layers.Dense(5, activation='softmax', name='OUTPUT')(x)

    #  建立模型
    model = tf.keras.Model(inputs=in_put, outputs=out_put, name=name)
    # 模型编译
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 建立模型
CNN_Model = build_cnn()
```


```python
CNN_Model.summary()
```

    Model: "CNN_V"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    INPUT (InputLayer)           [(None, 192, 192, 3)]     0         
    _________________________________________________________________
    CONV_1 (Conv2D)              (None, 190, 190, 32)      896       
    _________________________________________________________________
    CONV_2 (Conv2D)              (None, 188, 188, 64)      18496     
    _________________________________________________________________
    MAXPOOL_1 (MaxPooling2D)     (None, 94, 94, 64)        0         
    _________________________________________________________________
    CONV_3 (Conv2D)              (None, 94, 94, 64)        36928     
    _________________________________________________________________
    CONV_4 (Conv2D)              (None, 94, 94, 64)        36928     
    _________________________________________________________________
    MAXPOOL_2 (MaxPooling2D)     (None, 47, 47, 64)        0         
    _________________________________________________________________
    CONV_5 (Conv2D)              (None, 47, 47, 64)        36928     
    _________________________________________________________________
    CONV_6 (Conv2D)              (None, 47, 47, 64)        36928     
    _________________________________________________________________
    CONV_7 (Conv2D)              (None, 47, 47, 5)         325       
    _________________________________________________________________
    GAP_1 (GlobalAveragePooling2 (None, 5)                 0         
    _________________________________________________________________
    OUTPUT (Dense)               (None, 5)                 30        
    =================================================================
    Total params: 167,459
    Trainable params: 167,459
    Non-trainable params: 0
    _________________________________________________________________
    

+ ### 4.2 模型训练

建立回调，保存最佳模型。


```python
# 保存模型的文件夹
checkpoint_path_base = "./cnn_v-{val_accuracy:.5f}.ckpt"
checkpoint_dir_base = os.path.dirname(checkpoint_path_base)

# 创建一个回调，保证验证数据集损失最小
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_base, save_weights_only=True,
                                                    monitor='val_accuracy', mode='max', verbose=2, save_best_only=True)
```


```python
# 模型训练
model_history = CNN_Model.fit(train_ds, epochs=400, verbose=2, validation_data=val_ds, steps_per_epoch=10, callbacks=[model_callback])
```

    Train for 10 steps, validate for 13 steps
    Epoch 1/400
    
    Epoch 00001: val_accuracy improved from -inf to 0.21798, saving model to ./cnn_v-0.21798.ckpt
    10/10 - 165s - loss: 1.6065 - accuracy: 0.2234 - val_loss: 1.6072 - val_accuracy: 0.2180
    Epoch 2/400
    
    Epoch 00002: val_accuracy improved from 0.21798 to 0.24015, saving model to ./cnn_v-0.24015.ckpt
    10/10 - 129s - loss: 1.6085 - accuracy: 0.2516 - val_loss: 1.6073 - val_accuracy: 0.2401
    Epoch 3/400
    
    Epoch 00003: val_accuracy did not improve from 0.24015
    10/10 - 126s - loss: 1.6027 - accuracy: 0.2703 - val_loss: 1.6102 - val_accuracy: 0.2180
    Epoch 4/400
    
    Epoch 00004: val_accuracy improved from 0.24015 to 0.24384, saving model to ./cnn_v-0.24384.ckpt
    10/10 - 133s - loss: 1.6166 - accuracy: 0.1844 - val_loss: 1.6068 - val_accuracy: 0.2438
    Epoch 5/400
    
    Epoch 00005: val_accuracy did not improve from 0.24384
    10/10 - 129s - loss: 1.6063 - accuracy: 0.2625 - val_loss: 1.6059 - val_accuracy: 0.2438
    Epoch 6/400
    
    Epoch 00006: val_accuracy improved from 0.24384 to 0.25616, saving model to ./cnn_v-0.25616.ckpt
    10/10 - 128s - loss: 1.5925 - accuracy: 0.2750 - val_loss: 1.5430 - val_accuracy: 0.2562
    Epoch 7/400
    
    Epoch 00007: val_accuracy did not improve from 0.25616
    10/10 - 125s - loss: 1.4711 - accuracy: 0.2359 - val_loss: 1.4845 - val_accuracy: 0.2131
    Epoch 8/400
    
    Epoch 00008: val_accuracy did not improve from 0.25616
    10/10 - 124s - loss: 1.4569 - accuracy: 0.2062 - val_loss: 1.4821 - val_accuracy: 0.2217
    Epoch 9/400
    
    Epoch 00009: val_accuracy improved from 0.25616 to 0.29803, saving model to ./cnn_v-0.29803.ckpt
    10/10 - 125s - loss: 1.4831 - accuracy: 0.2828 - val_loss: 1.4605 - val_accuracy: 0.2980
    Epoch 10/400
    
    Epoch 00010: val_accuracy did not improve from 0.29803
    10/10 - 130s - loss: 1.4244 - accuracy: 0.2266 - val_loss: 1.4309 - val_accuracy: 0.2857
    Epoch 11/400
    
    Epoch 00011: val_accuracy did not improve from 0.29803
    10/10 - 138s - loss: 1.3920 - accuracy: 0.2906 - val_loss: 1.4186 - val_accuracy: 0.2845
    Epoch 12/400
    
    Epoch 00012: val_accuracy did not improve from 0.29803
    10/10 - 133s - loss: 1.3742 - accuracy: 0.2953 - val_loss: 1.4322 - val_accuracy: 0.2906
    Epoch 13/400
    
    Epoch 00013: val_accuracy improved from 0.29803 to 0.32389, saving model to ./cnn_v-0.32389.ckpt
    10/10 - 129s - loss: 1.4161 - accuracy: 0.3109 - val_loss: 1.4210 - val_accuracy: 0.3239
    Epoch 14/400
    
    Epoch 00014: val_accuracy improved from 0.32389 to 0.33251, saving model to ./cnn_v-0.33251.ckpt
    10/10 - 122s - loss: 1.3599 - accuracy: 0.3375 - val_loss: 1.3908 - val_accuracy: 0.3325
    Epoch 15/400
    
    Epoch 00015: val_accuracy did not improve from 0.33251
    10/10 - 126s - loss: 1.3700 - accuracy: 0.3969 - val_loss: 1.3929 - val_accuracy: 0.3202
    Epoch 16/400
    
    Epoch 00016: val_accuracy improved from 0.33251 to 0.34236, saving model to ./cnn_v-0.34236.ckpt
    10/10 - 121s - loss: 1.3724 - accuracy: 0.3656 - val_loss: 1.3808 - val_accuracy: 0.3424
    Epoch 17/400
    
    Epoch 00017: val_accuracy improved from 0.34236 to 0.39409, saving model to ./cnn_v-0.39409.ckpt
    10/10 - 121s - loss: 1.3485 - accuracy: 0.3688 - val_loss: 1.3791 - val_accuracy: 0.3941
    Epoch 18/400
    
    Epoch 00018: val_accuracy did not improve from 0.39409
    10/10 - 121s - loss: 1.3550 - accuracy: 0.4156 - val_loss: 1.3777 - val_accuracy: 0.3695
    Epoch 19/400
    
    Epoch 00019: val_accuracy did not improve from 0.39409
    10/10 - 120s - loss: 1.3390 - accuracy: 0.4328 - val_loss: 1.3999 - val_accuracy: 0.3645
    Epoch 20/400
    
    Epoch 00020: val_accuracy improved from 0.39409 to 0.39532, saving model to ./cnn_v-0.39532.ckpt
    10/10 - 120s - loss: 1.3271 - accuracy: 0.4375 - val_loss: 1.4132 - val_accuracy: 0.3953
    Epoch 21/400
    
    Epoch 00021: val_accuracy did not improve from 0.39532
    10/10 - 120s - loss: 1.3829 - accuracy: 0.3891 - val_loss: 1.3826 - val_accuracy: 0.3953
    Epoch 22/400
    
    Epoch 00022: val_accuracy improved from 0.39532 to 0.41010, saving model to ./cnn_v-0.41010.ckpt
    10/10 - 120s - loss: 1.3310 - accuracy: 0.4437 - val_loss: 1.3617 - val_accuracy: 0.4101
    Epoch 23/400
    
    Epoch 00023: val_accuracy improved from 0.41010 to 0.41626, saving model to ./cnn_v-0.41626.ckpt
    10/10 - 121s - loss: 1.2811 - accuracy: 0.4781 - val_loss: 1.3393 - val_accuracy: 0.4163
    Epoch 24/400
    
    Epoch 00024: val_accuracy did not improve from 0.41626
    10/10 - 120s - loss: 1.2874 - accuracy: 0.4688 - val_loss: 1.3503 - val_accuracy: 0.4064
    Epoch 25/400
    
    Epoch 00025: val_accuracy did not improve from 0.41626
    10/10 - 119s - loss: 1.3002 - accuracy: 0.4469 - val_loss: 1.3459 - val_accuracy: 0.4076
    Epoch 26/400
    
    Epoch 00026: val_accuracy did not improve from 0.41626
    10/10 - 120s - loss: 1.3213 - accuracy: 0.4234 - val_loss: 1.3719 - val_accuracy: 0.4113
    Epoch 27/400
    
    Epoch 00027: val_accuracy did not improve from 0.41626
    10/10 - 120s - loss: 1.3105 - accuracy: 0.4297 - val_loss: 1.3419 - val_accuracy: 0.3892
    Epoch 28/400
    
    Epoch 00028: val_accuracy improved from 0.41626 to 0.46059, saving model to ./cnn_v-0.46059.ckpt
    10/10 - 120s - loss: 1.2779 - accuracy: 0.4437 - val_loss: 1.3185 - val_accuracy: 0.4606
    Epoch 29/400
    
    Epoch 00029: val_accuracy did not improve from 0.46059
    10/10 - 120s - loss: 1.2749 - accuracy: 0.4688 - val_loss: 1.3217 - val_accuracy: 0.4101
    Epoch 30/400
    
    Epoch 00030: val_accuracy did not improve from 0.46059
    10/10 - 120s - loss: 1.2381 - accuracy: 0.4656 - val_loss: 1.3014 - val_accuracy: 0.4384
    Epoch 31/400
    
    Epoch 00031: val_accuracy did not improve from 0.46059
    10/10 - 119s - loss: 1.2901 - accuracy: 0.4563 - val_loss: 1.3060 - val_accuracy: 0.4372
    Epoch 32/400
    
    Epoch 00032: val_accuracy improved from 0.46059 to 0.46182, saving model to ./cnn_v-0.46182.ckpt
    10/10 - 120s - loss: 1.2173 - accuracy: 0.5172 - val_loss: 1.2990 - val_accuracy: 0.4618
    Epoch 33/400
    
    Epoch 00033: val_accuracy improved from 0.46182 to 0.47167, saving model to ./cnn_v-0.47167.ckpt
    10/10 - 120s - loss: 1.2473 - accuracy: 0.4844 - val_loss: 1.2921 - val_accuracy: 0.4717
    Epoch 34/400
    
    Epoch 00034: val_accuracy did not improve from 0.47167
    10/10 - 121s - loss: 1.2637 - accuracy: 0.5172 - val_loss: 1.3308 - val_accuracy: 0.4101
    Epoch 35/400
    
    Epoch 00035: val_accuracy did not improve from 0.47167
    10/10 - 119s - loss: 1.2586 - accuracy: 0.4875 - val_loss: 1.2815 - val_accuracy: 0.4631
    Epoch 36/400
    
    Epoch 00036: val_accuracy improved from 0.47167 to 0.48276, saving model to ./cnn_v-0.48276.ckpt
    10/10 - 120s - loss: 1.1686 - accuracy: 0.5172 - val_loss: 1.2672 - val_accuracy: 0.4828
    Epoch 37/400
    
    Epoch 00037: val_accuracy did not improve from 0.48276
    10/10 - 120s - loss: 1.2563 - accuracy: 0.4953 - val_loss: 1.2552 - val_accuracy: 0.4606
    Epoch 38/400
    
    Epoch 00038: val_accuracy did not improve from 0.48276
    10/10 - 120s - loss: 1.1668 - accuracy: 0.5141 - val_loss: 1.2389 - val_accuracy: 0.4470
    Epoch 39/400
    
    Epoch 00039: val_accuracy did not improve from 0.48276
    10/10 - 121s - loss: 1.2423 - accuracy: 0.4906 - val_loss: 1.2576 - val_accuracy: 0.4557
    Epoch 40/400
    
    Epoch 00040: val_accuracy improved from 0.48276 to 0.48399, saving model to ./cnn_v-0.48399.ckpt
    10/10 - 120s - loss: 1.1360 - accuracy: 0.5359 - val_loss: 1.2451 - val_accuracy: 0.4840
    Epoch 41/400
    
    Epoch 00041: val_accuracy improved from 0.48399 to 0.49631, saving model to ./cnn_v-0.49631.ckpt
    10/10 - 120s - loss: 1.1341 - accuracy: 0.5094 - val_loss: 1.1752 - val_accuracy: 0.4963
    Epoch 42/400
    
    Epoch 00042: val_accuracy did not improve from 0.49631
    10/10 - 119s - loss: 1.1352 - accuracy: 0.5406 - val_loss: 1.2183 - val_accuracy: 0.4865
    Epoch 43/400
    
    Epoch 00043: val_accuracy improved from 0.49631 to 0.51847, saving model to ./cnn_v-0.51847.ckpt
    10/10 - 120s - loss: 1.0947 - accuracy: 0.5891 - val_loss: 1.1485 - val_accuracy: 0.5185
    Epoch 44/400
    
    Epoch 00044: val_accuracy did not improve from 0.51847
    10/10 - 120s - loss: 1.0888 - accuracy: 0.5391 - val_loss: 1.1433 - val_accuracy: 0.5172
    Epoch 45/400
    
    Epoch 00045: val_accuracy improved from 0.51847 to 0.52340, saving model to ./cnn_v-0.52340.ckpt
    10/10 - 120s - loss: 1.0474 - accuracy: 0.5484 - val_loss: 1.1326 - val_accuracy: 0.5234
    Epoch 46/400
    
    Epoch 00046: val_accuracy did not improve from 0.52340
    10/10 - 120s - loss: 1.0608 - accuracy: 0.5562 - val_loss: 1.1390 - val_accuracy: 0.5160
    Epoch 47/400
    
    Epoch 00047: val_accuracy did not improve from 0.52340
    10/10 - 120s - loss: 1.0457 - accuracy: 0.5594 - val_loss: 1.1300 - val_accuracy: 0.5160
    Epoch 48/400
    
    Epoch 00048: val_accuracy improved from 0.52340 to 0.53941, saving model to ./cnn_v-0.53941.ckpt
    10/10 - 120s - loss: 1.0172 - accuracy: 0.5750 - val_loss: 1.1193 - val_accuracy: 0.5394
    Epoch 49/400
    
    Epoch 00049: val_accuracy did not improve from 0.53941
    10/10 - 120s - loss: 1.0317 - accuracy: 0.5609 - val_loss: 1.1610 - val_accuracy: 0.5296
    Epoch 50/400
    
    Epoch 00050: val_accuracy did not improve from 0.53941
    10/10 - 119s - loss: 1.0730 - accuracy: 0.5719 - val_loss: 1.1122 - val_accuracy: 0.5357
    Epoch 51/400
    
    Epoch 00051: val_accuracy did not improve from 0.53941
    10/10 - 122s - loss: 1.0753 - accuracy: 0.5531 - val_loss: 1.1274 - val_accuracy: 0.5135
    Epoch 52/400
    
    Epoch 00052: val_accuracy improved from 0.53941 to 0.55665, saving model to ./cnn_v-0.55665.ckpt
    10/10 - 120s - loss: 1.0533 - accuracy: 0.5437 - val_loss: 1.0955 - val_accuracy: 0.5567
    Epoch 53/400
    
    Epoch 00053: val_accuracy did not improve from 0.55665
    10/10 - 120s - loss: 0.9807 - accuracy: 0.5969 - val_loss: 1.1339 - val_accuracy: 0.5185
    Epoch 54/400
    
    Epoch 00054: val_accuracy improved from 0.55665 to 0.58251, saving model to ./cnn_v-0.58251.ckpt
    10/10 - 120s - loss: 0.9737 - accuracy: 0.6078 - val_loss: 1.0647 - val_accuracy: 0.5825
    Epoch 55/400
    
    Epoch 00055: val_accuracy did not improve from 0.58251
    10/10 - 120s - loss: 0.9927 - accuracy: 0.6000 - val_loss: 1.1103 - val_accuracy: 0.5530
    Epoch 56/400
    
    Epoch 00056: val_accuracy did not improve from 0.58251
    10/10 - 120s - loss: 1.0227 - accuracy: 0.6125 - val_loss: 1.0664 - val_accuracy: 0.5653
    Epoch 57/400
    
    Epoch 00057: val_accuracy did not improve from 0.58251
    10/10 - 120s - loss: 1.0120 - accuracy: 0.5859 - val_loss: 1.0876 - val_accuracy: 0.5456
    Epoch 58/400
    
    Epoch 00058: val_accuracy did not improve from 0.58251
    10/10 - 119s - loss: 0.9726 - accuracy: 0.5891 - val_loss: 1.0468 - val_accuracy: 0.5751
    Epoch 59/400
    
    Epoch 00059: val_accuracy did not improve from 0.58251
    10/10 - 120s - loss: 0.9870 - accuracy: 0.5906 - val_loss: 1.0461 - val_accuracy: 0.5591
    Epoch 60/400
    
    Epoch 00060: val_accuracy improved from 0.58251 to 0.60345, saving model to ./cnn_v-0.60345.ckpt
    10/10 - 119s - loss: 0.9629 - accuracy: 0.6266 - val_loss: 1.0317 - val_accuracy: 0.6034
    Epoch 61/400
    
    Epoch 00061: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 0.9233 - accuracy: 0.6406 - val_loss: 1.0606 - val_accuracy: 0.5751
    Epoch 62/400
    
    Epoch 00062: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 0.9635 - accuracy: 0.5844 - val_loss: 1.0401 - val_accuracy: 0.5591
    Epoch 63/400
    
    Epoch 00063: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 0.8857 - accuracy: 0.6562 - val_loss: 1.0529 - val_accuracy: 0.5948
    Epoch 64/400
    
    Epoch 00064: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 1.0159 - accuracy: 0.6031 - val_loss: 1.0296 - val_accuracy: 0.5837
    Epoch 65/400
    
    Epoch 00065: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 0.9845 - accuracy: 0.6172 - val_loss: 1.0733 - val_accuracy: 0.5554
    Epoch 66/400
    
    Epoch 00066: val_accuracy did not improve from 0.60345
    10/10 - 120s - loss: 0.9084 - accuracy: 0.6359 - val_loss: 1.0168 - val_accuracy: 0.5973
    Epoch 67/400
    
    Epoch 00067: val_accuracy improved from 0.60345 to 0.60961, saving model to ./cnn_v-0.60961.ckpt
    10/10 - 120s - loss: 0.9167 - accuracy: 0.6187 - val_loss: 1.0129 - val_accuracy: 0.6096
    Epoch 68/400
    
    Epoch 00068: val_accuracy did not improve from 0.60961
    10/10 - 119s - loss: 0.8993 - accuracy: 0.6687 - val_loss: 1.0113 - val_accuracy: 0.5948
    Epoch 69/400
    
    Epoch 00069: val_accuracy improved from 0.60961 to 0.62438, saving model to ./cnn_v-0.62438.ckpt
    10/10 - 120s - loss: 0.9479 - accuracy: 0.6156 - val_loss: 0.9604 - val_accuracy: 0.6244
    Epoch 70/400
    
    Epoch 00070: val_accuracy did not improve from 0.62438
    10/10 - 119s - loss: 0.8620 - accuracy: 0.6734 - val_loss: 0.9888 - val_accuracy: 0.6158
    Epoch 71/400
    
    Epoch 00071: val_accuracy did not improve from 0.62438
    10/10 - 119s - loss: 0.8874 - accuracy: 0.6594 - val_loss: 0.9815 - val_accuracy: 0.6059
    Epoch 72/400
    
    Epoch 00072: val_accuracy did not improve from 0.62438
    10/10 - 120s - loss: 0.8898 - accuracy: 0.6562 - val_loss: 0.9909 - val_accuracy: 0.6158
    Epoch 73/400
    
    Epoch 00073: val_accuracy did not improve from 0.62438
    10/10 - 120s - loss: 0.9073 - accuracy: 0.6453 - val_loss: 0.9500 - val_accuracy: 0.6244
    Epoch 74/400
    
    Epoch 00074: val_accuracy did not improve from 0.62438
    10/10 - 119s - loss: 0.8729 - accuracy: 0.6547 - val_loss: 1.0160 - val_accuracy: 0.6084
    Epoch 75/400
    
    Epoch 00075: val_accuracy did not improve from 0.62438
    10/10 - 120s - loss: 0.9125 - accuracy: 0.6406 - val_loss: 0.9506 - val_accuracy: 0.6219
    Epoch 76/400
    
    Epoch 00076: val_accuracy did not improve from 0.62438
    10/10 - 119s - loss: 0.8539 - accuracy: 0.6750 - val_loss: 0.9758 - val_accuracy: 0.6084
    Epoch 77/400
    
    Epoch 00077: val_accuracy did not improve from 0.62438
    10/10 - 120s - loss: 0.8157 - accuracy: 0.7016 - val_loss: 1.0176 - val_accuracy: 0.5924
    Epoch 78/400
    
    Epoch 00078: val_accuracy did not improve from 0.62438
    10/10 - 119s - loss: 0.8194 - accuracy: 0.6797 - val_loss: 1.0588 - val_accuracy: 0.5862
    Epoch 79/400
    
    Epoch 00079: val_accuracy improved from 0.62438 to 0.62562, saving model to ./cnn_v-0.62562.ckpt
    10/10 - 120s - loss: 0.9549 - accuracy: 0.6422 - val_loss: 0.9720 - val_accuracy: 0.6256
    Epoch 80/400
    
    Epoch 00080: val_accuracy improved from 0.62562 to 0.64409, saving model to ./cnn_v-0.64409.ckpt
    10/10 - 120s - loss: 0.8576 - accuracy: 0.6594 - val_loss: 0.9280 - val_accuracy: 0.6441
    Epoch 81/400
    
    Epoch 00081: val_accuracy did not improve from 0.64409
    10/10 - 120s - loss: 0.7990 - accuracy: 0.7063 - val_loss: 0.9369 - val_accuracy: 0.6404
    Epoch 82/400
    
    Epoch 00082: val_accuracy did not improve from 0.64409
    10/10 - 119s - loss: 0.8241 - accuracy: 0.6781 - val_loss: 0.9406 - val_accuracy: 0.6256
    Epoch 83/400
    
    Epoch 00083: val_accuracy did not improve from 0.64409
    10/10 - 120s - loss: 0.8555 - accuracy: 0.6656 - val_loss: 0.9064 - val_accuracy: 0.6404
    Epoch 84/400
    
    Epoch 00084: val_accuracy did not improve from 0.64409
    10/10 - 119s - loss: 0.8203 - accuracy: 0.6844 - val_loss: 0.9738 - val_accuracy: 0.6244
    Epoch 85/400
    
    Epoch 00085: val_accuracy did not improve from 0.64409
    10/10 - 120s - loss: 0.8310 - accuracy: 0.6859 - val_loss: 0.9527 - val_accuracy: 0.6207
    Epoch 86/400
    
    Epoch 00086: val_accuracy improved from 0.64409 to 0.65517, saving model to ./cnn_v-0.65517.ckpt
    10/10 - 120s - loss: 0.8366 - accuracy: 0.6797 - val_loss: 0.9043 - val_accuracy: 0.6552
    Epoch 87/400
    
    Epoch 00087: val_accuracy did not improve from 0.65517
    10/10 - 119s - loss: 0.8873 - accuracy: 0.6578 - val_loss: 0.9669 - val_accuracy: 0.6466
    Epoch 88/400
    
    Epoch 00088: val_accuracy did not improve from 0.65517
    10/10 - 119s - loss: 0.8438 - accuracy: 0.6844 - val_loss: 0.9411 - val_accuracy: 0.6305
    Epoch 89/400
    
    Epoch 00089: val_accuracy improved from 0.65517 to 0.66749, saving model to ./cnn_v-0.66749.ckpt
    10/10 - 120s - loss: 0.7125 - accuracy: 0.7406 - val_loss: 0.8917 - val_accuracy: 0.6675
    Epoch 90/400
    
    Epoch 00090: val_accuracy did not improve from 0.66749
    10/10 - 120s - loss: 0.8200 - accuracy: 0.6875 - val_loss: 0.8854 - val_accuracy: 0.6527
    Epoch 91/400
    
    Epoch 00091: val_accuracy did not improve from 0.66749
    10/10 - 119s - loss: 0.7922 - accuracy: 0.7016 - val_loss: 0.9781 - val_accuracy: 0.5948
    Epoch 92/400
    
    Epoch 00092: val_accuracy did not improve from 0.66749
    10/10 - 120s - loss: 0.8688 - accuracy: 0.6656 - val_loss: 0.9295 - val_accuracy: 0.6527
    Epoch 93/400
    
    Epoch 00093: val_accuracy did not improve from 0.66749
    10/10 - 119s - loss: 0.8235 - accuracy: 0.6781 - val_loss: 0.9445 - val_accuracy: 0.6601
    Epoch 94/400
    
    Epoch 00094: val_accuracy did not improve from 0.66749
    10/10 - 119s - loss: 0.7642 - accuracy: 0.7312 - val_loss: 0.9082 - val_accuracy: 0.6478
    Epoch 95/400
    
    Epoch 00095: val_accuracy improved from 0.66749 to 0.66872, saving model to ./cnn_v-0.66872.ckpt
    10/10 - 120s - loss: 0.7370 - accuracy: 0.7156 - val_loss: 0.8888 - val_accuracy: 0.6687
    Epoch 96/400
    
    Epoch 00096: val_accuracy did not improve from 0.66872
    10/10 - 120s - loss: 0.8228 - accuracy: 0.6922 - val_loss: 0.8839 - val_accuracy: 0.6626
    Epoch 97/400
    
    Epoch 00097: val_accuracy improved from 0.66872 to 0.66995, saving model to ./cnn_v-0.66995.ckpt
    10/10 - 120s - loss: 0.7071 - accuracy: 0.7406 - val_loss: 0.8902 - val_accuracy: 0.6700
    Epoch 98/400
    
    Epoch 00098: val_accuracy did not improve from 0.66995
    10/10 - 120s - loss: 0.7842 - accuracy: 0.7125 - val_loss: 0.8777 - val_accuracy: 0.6515
    Epoch 99/400
    
    Epoch 00099: val_accuracy improved from 0.66995 to 0.68842, saving model to ./cnn_v-0.68842.ckpt
    10/10 - 119s - loss: 0.7596 - accuracy: 0.7188 - val_loss: 0.8623 - val_accuracy: 0.6884
    Epoch 100/400
    
    Epoch 00100: val_accuracy improved from 0.68842 to 0.69335, saving model to ./cnn_v-0.69335.ckpt
    10/10 - 120s - loss: 0.7129 - accuracy: 0.7344 - val_loss: 0.8421 - val_accuracy: 0.6933
    Epoch 101/400
    
    Epoch 00101: val_accuracy did not improve from 0.69335
    10/10 - 119s - loss: 0.6970 - accuracy: 0.7406 - val_loss: 0.8431 - val_accuracy: 0.6884
    Epoch 102/400
    
    Epoch 00102: val_accuracy improved from 0.69335 to 0.69458, saving model to ./cnn_v-0.69458.ckpt
    10/10 - 119s - loss: 0.7832 - accuracy: 0.7078 - val_loss: 0.8539 - val_accuracy: 0.6946
    Epoch 103/400
    
    Epoch 00103: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.6742 - accuracy: 0.7453 - val_loss: 0.8419 - val_accuracy: 0.6810
    Epoch 104/400
    
    Epoch 00104: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7592 - accuracy: 0.7125 - val_loss: 0.8638 - val_accuracy: 0.6552
    Epoch 105/400
    
    Epoch 00105: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7049 - accuracy: 0.7469 - val_loss: 0.8103 - val_accuracy: 0.6884
    Epoch 106/400
    
    Epoch 00106: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.6631 - accuracy: 0.7500 - val_loss: 0.8748 - val_accuracy: 0.6810
    Epoch 107/400
    
    Epoch 00107: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7342 - accuracy: 0.7375 - val_loss: 0.8271 - val_accuracy: 0.6835
    Epoch 108/400
    
    Epoch 00108: val_accuracy did not improve from 0.69458
    10/10 - 120s - loss: 0.7384 - accuracy: 0.7234 - val_loss: 0.8465 - val_accuracy: 0.6675
    Epoch 109/400
    
    Epoch 00109: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7032 - accuracy: 0.7641 - val_loss: 0.8442 - val_accuracy: 0.6712
    Epoch 110/400
    
    Epoch 00110: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7654 - accuracy: 0.7172 - val_loss: 0.8394 - val_accuracy: 0.6884
    Epoch 111/400
    
    Epoch 00111: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.6967 - accuracy: 0.7188 - val_loss: 0.8446 - val_accuracy: 0.6872
    Epoch 112/400
    
    Epoch 00112: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.6592 - accuracy: 0.7672 - val_loss: 0.8619 - val_accuracy: 0.6810
    Epoch 113/400
    
    Epoch 00113: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.6576 - accuracy: 0.7625 - val_loss: 0.8434 - val_accuracy: 0.6823
    Epoch 114/400
    
    Epoch 00114: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7102 - accuracy: 0.7312 - val_loss: 0.8577 - val_accuracy: 0.6638
    Epoch 115/400
    
    Epoch 00115: val_accuracy did not improve from 0.69458
    10/10 - 119s - loss: 0.7441 - accuracy: 0.7312 - val_loss: 0.8511 - val_accuracy: 0.6724
    Epoch 116/400
    
    Epoch 00116: val_accuracy did not improve from 0.69458
    10/10 - 120s - loss: 0.6901 - accuracy: 0.7469 - val_loss: 0.8238 - val_accuracy: 0.6884
    Epoch 117/400
    
    Epoch 00117: val_accuracy improved from 0.69458 to 0.69704, saving model to ./cnn_v-0.69704.ckpt
    10/10 - 120s - loss: 0.6234 - accuracy: 0.7875 - val_loss: 0.8161 - val_accuracy: 0.6970
    Epoch 118/400
    
    Epoch 00118: val_accuracy improved from 0.69704 to 0.69951, saving model to ./cnn_v-0.69951.ckpt
    10/10 - 119s - loss: 0.6945 - accuracy: 0.7281 - val_loss: 0.7942 - val_accuracy: 0.6995
    Epoch 119/400
    
    Epoch 00119: val_accuracy did not improve from 0.69951
    10/10 - 119s - loss: 0.6733 - accuracy: 0.7469 - val_loss: 0.7907 - val_accuracy: 0.6860
    Epoch 120/400
    
    Epoch 00120: val_accuracy did not improve from 0.69951
    10/10 - 119s - loss: 0.6556 - accuracy: 0.7625 - val_loss: 0.8243 - val_accuracy: 0.6749
    Epoch 121/400
    
    Epoch 00121: val_accuracy did not improve from 0.69951
    10/10 - 119s - loss: 0.6414 - accuracy: 0.7812 - val_loss: 0.8148 - val_accuracy: 0.6946
    Epoch 122/400
    
    Epoch 00122: val_accuracy did not improve from 0.69951
    10/10 - 119s - loss: 0.5976 - accuracy: 0.7766 - val_loss: 0.8696 - val_accuracy: 0.6773
    Epoch 123/400
    
    Epoch 00123: val_accuracy improved from 0.69951 to 0.70936, saving model to ./cnn_v-0.70936.ckpt
    10/10 - 119s - loss: 0.6913 - accuracy: 0.7266 - val_loss: 0.7905 - val_accuracy: 0.7094
    Epoch 124/400
    
    Epoch 00124: val_accuracy did not improve from 0.70936
    10/10 - 119s - loss: 0.7120 - accuracy: 0.7219 - val_loss: 0.8451 - val_accuracy: 0.6921
    Epoch 125/400
    
    Epoch 00125: val_accuracy did not improve from 0.70936
    10/10 - 119s - loss: 0.6103 - accuracy: 0.7812 - val_loss: 0.8166 - val_accuracy: 0.6921
    Epoch 126/400
    
    Epoch 00126: val_accuracy did not improve from 0.70936
    10/10 - 121s - loss: 0.6822 - accuracy: 0.7656 - val_loss: 0.8013 - val_accuracy: 0.6909
    Epoch 127/400
    
    Epoch 00127: val_accuracy improved from 0.70936 to 0.71059, saving model to ./cnn_v-0.71059.ckpt
    10/10 - 120s - loss: 0.6347 - accuracy: 0.7531 - val_loss: 0.7733 - val_accuracy: 0.7106
    Epoch 128/400
    
    Epoch 00128: val_accuracy improved from 0.71059 to 0.71429, saving model to ./cnn_v-0.71429.ckpt
    10/10 - 119s - loss: 0.6097 - accuracy: 0.7781 - val_loss: 0.7869 - val_accuracy: 0.7143
    Epoch 129/400
    
    Epoch 00129: val_accuracy did not improve from 0.71429
    10/10 - 119s - loss: 0.6159 - accuracy: 0.7812 - val_loss: 0.7787 - val_accuracy: 0.7007
    Epoch 130/400
    
    Epoch 00130: val_accuracy improved from 0.71429 to 0.71675, saving model to ./cnn_v-0.71675.ckpt
    10/10 - 119s - loss: 0.6266 - accuracy: 0.7797 - val_loss: 0.7570 - val_accuracy: 0.7167
    Epoch 131/400
    
    Epoch 00131: val_accuracy did not improve from 0.71675
    10/10 - 119s - loss: 0.5493 - accuracy: 0.7937 - val_loss: 0.7667 - val_accuracy: 0.7131
    Epoch 132/400
    
    Epoch 00132: val_accuracy did not improve from 0.71675
    10/10 - 119s - loss: 0.6111 - accuracy: 0.7812 - val_loss: 0.7539 - val_accuracy: 0.7167
    Epoch 133/400
    
    Epoch 00133: val_accuracy did not improve from 0.71675
    10/10 - 119s - loss: 0.5803 - accuracy: 0.8016 - val_loss: 0.8086 - val_accuracy: 0.7020
    Epoch 134/400
    
    Epoch 00134: val_accuracy improved from 0.71675 to 0.72906, saving model to ./cnn_v-0.72906.ckpt
    10/10 - 119s - loss: 0.6301 - accuracy: 0.7766 - val_loss: 0.7447 - val_accuracy: 0.7291
    Epoch 135/400
    
    Epoch 00135: val_accuracy did not improve from 0.72906
    10/10 - 119s - loss: 0.5896 - accuracy: 0.8016 - val_loss: 0.7775 - val_accuracy: 0.7192
    Epoch 136/400
    
    Epoch 00136: val_accuracy did not improve from 0.72906
    10/10 - 119s - loss: 0.6121 - accuracy: 0.7859 - val_loss: 0.7967 - val_accuracy: 0.7155
    Epoch 137/400
    
    Epoch 00137: val_accuracy did not improve from 0.72906
    10/10 - 119s - loss: 0.6364 - accuracy: 0.7688 - val_loss: 0.7679 - val_accuracy: 0.7007
    Epoch 138/400
    
    Epoch 00138: val_accuracy improved from 0.72906 to 0.73030, saving model to ./cnn_v-0.73030.ckpt
    10/10 - 119s - loss: 0.5903 - accuracy: 0.7891 - val_loss: 0.7341 - val_accuracy: 0.7303
    Epoch 139/400
    
    Epoch 00139: val_accuracy did not improve from 0.73030
    10/10 - 120s - loss: 0.5759 - accuracy: 0.7766 - val_loss: 0.8097 - val_accuracy: 0.7143
    Epoch 140/400
    
    Epoch 00140: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.6550 - accuracy: 0.7719 - val_loss: 0.7425 - val_accuracy: 0.7303
    Epoch 141/400
    
    Epoch 00141: val_accuracy did not improve from 0.73030
    10/10 - 120s - loss: 0.5740 - accuracy: 0.7922 - val_loss: 0.7516 - val_accuracy: 0.7069
    Epoch 142/400
    
    Epoch 00142: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.5655 - accuracy: 0.7953 - val_loss: 0.7265 - val_accuracy: 0.7204
    Epoch 143/400
    
    Epoch 00143: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.5386 - accuracy: 0.8047 - val_loss: 0.7316 - val_accuracy: 0.7266
    Epoch 144/400
    
    Epoch 00144: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.5702 - accuracy: 0.8062 - val_loss: 0.7370 - val_accuracy: 0.7254
    Epoch 145/400
    
    Epoch 00145: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.5667 - accuracy: 0.7859 - val_loss: 0.7037 - val_accuracy: 0.7254
    Epoch 146/400
    
    Epoch 00146: val_accuracy did not improve from 0.73030
    10/10 - 119s - loss: 0.6263 - accuracy: 0.7672 - val_loss: 0.7747 - val_accuracy: 0.7044
    Epoch 147/400
    
    Epoch 00147: val_accuracy improved from 0.73030 to 0.73276, saving model to ./cnn_v-0.73276.ckpt
    10/10 - 119s - loss: 0.5920 - accuracy: 0.7953 - val_loss: 0.7154 - val_accuracy: 0.7328
    Epoch 148/400
    
    Epoch 00148: val_accuracy did not improve from 0.73276
    10/10 - 132s - loss: 0.5815 - accuracy: 0.8047 - val_loss: 0.7301 - val_accuracy: 0.7180
    Epoch 149/400
    
    Epoch 00149: val_accuracy did not improve from 0.73276
    10/10 - 119s - loss: 0.5472 - accuracy: 0.8250 - val_loss: 0.7102 - val_accuracy: 0.7291
    Epoch 150/400
    
    Epoch 00150: val_accuracy improved from 0.73276 to 0.73645, saving model to ./cnn_v-0.73645.ckpt
    10/10 - 119s - loss: 0.5692 - accuracy: 0.8031 - val_loss: 0.7220 - val_accuracy: 0.7365
    Epoch 151/400
    
    Epoch 00151: val_accuracy did not improve from 0.73645
    10/10 - 120s - loss: 0.5465 - accuracy: 0.8047 - val_loss: 0.7477 - val_accuracy: 0.7328
    Epoch 152/400
    
    Epoch 00152: val_accuracy did not improve from 0.73645
    10/10 - 119s - loss: 0.5520 - accuracy: 0.8156 - val_loss: 0.7625 - val_accuracy: 0.7315
    Epoch 153/400
    
    Epoch 00153: val_accuracy did not improve from 0.73645
    10/10 - 119s - loss: 0.5963 - accuracy: 0.7781 - val_loss: 0.7429 - val_accuracy: 0.7155
    Epoch 154/400
    
    Epoch 00154: val_accuracy improved from 0.73645 to 0.73892, saving model to ./cnn_v-0.73892.ckpt
    10/10 - 119s - loss: 0.5559 - accuracy: 0.8016 - val_loss: 0.7226 - val_accuracy: 0.7389
    Epoch 155/400
    
    Epoch 00155: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.4940 - accuracy: 0.8234 - val_loss: 0.7361 - val_accuracy: 0.7328
    Epoch 156/400
    
    Epoch 00156: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.5153 - accuracy: 0.8125 - val_loss: 0.7292 - val_accuracy: 0.7352
    Epoch 157/400
    
    Epoch 00157: val_accuracy did not improve from 0.73892
    10/10 - 120s - loss: 0.5674 - accuracy: 0.7922 - val_loss: 0.7707 - val_accuracy: 0.7254
    Epoch 158/400
    
    Epoch 00158: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.5352 - accuracy: 0.7937 - val_loss: 0.7508 - val_accuracy: 0.7192
    Epoch 159/400
    
    Epoch 00159: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.5397 - accuracy: 0.7953 - val_loss: 0.7115 - val_accuracy: 0.7315
    Epoch 160/400
    
    Epoch 00160: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.6642 - accuracy: 0.7766 - val_loss: 0.7662 - val_accuracy: 0.7217
    Epoch 161/400
    
    Epoch 00161: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.6380 - accuracy: 0.7859 - val_loss: 0.7675 - val_accuracy: 0.7155
    Epoch 162/400
    
    Epoch 00162: val_accuracy did not improve from 0.73892
    10/10 - 120s - loss: 0.6092 - accuracy: 0.7812 - val_loss: 0.7239 - val_accuracy: 0.7217
    Epoch 163/400
    
    Epoch 00163: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.5993 - accuracy: 0.7891 - val_loss: 0.6979 - val_accuracy: 0.7328
    Epoch 164/400
    
    Epoch 00164: val_accuracy did not improve from 0.73892
    10/10 - 119s - loss: 0.4537 - accuracy: 0.8391 - val_loss: 0.7394 - val_accuracy: 0.7254
    Epoch 165/400
    
    Epoch 00165: val_accuracy improved from 0.73892 to 0.74877, saving model to ./cnn_v-0.74877.ckpt
    10/10 - 119s - loss: 0.5238 - accuracy: 0.7984 - val_loss: 0.7211 - val_accuracy: 0.7488
    Epoch 166/400
    
    Epoch 00166: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.4694 - accuracy: 0.8422 - val_loss: 0.7270 - val_accuracy: 0.7328
    Epoch 167/400
    
    Epoch 00167: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.5262 - accuracy: 0.8094 - val_loss: 0.7808 - val_accuracy: 0.7241
    Epoch 168/400
    
    Epoch 00168: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.5572 - accuracy: 0.8047 - val_loss: 0.7351 - val_accuracy: 0.7315
    Epoch 169/400
    
    Epoch 00169: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.5821 - accuracy: 0.7859 - val_loss: 0.7779 - val_accuracy: 0.7155
    Epoch 170/400
    
    Epoch 00170: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.4346 - accuracy: 0.8469 - val_loss: 0.7054 - val_accuracy: 0.7488
    Epoch 171/400
    
    Epoch 00171: val_accuracy did not improve from 0.74877
    10/10 - 119s - loss: 0.5059 - accuracy: 0.8344 - val_loss: 0.7167 - val_accuracy: 0.7278
    Epoch 172/400
    
    Epoch 00172: val_accuracy improved from 0.74877 to 0.75123, saving model to ./cnn_v-0.75123.ckpt
    10/10 - 120s - loss: 0.5102 - accuracy: 0.8078 - val_loss: 0.6625 - val_accuracy: 0.7512
    Epoch 173/400
    
    Epoch 00173: val_accuracy did not improve from 0.75123
    10/10 - 119s - loss: 0.5840 - accuracy: 0.8031 - val_loss: 0.7372 - val_accuracy: 0.7426
    Epoch 174/400
    
    Epoch 00174: val_accuracy did not improve from 0.75123
    10/10 - 119s - loss: 0.5119 - accuracy: 0.8078 - val_loss: 0.6929 - val_accuracy: 0.7426
    Epoch 175/400
    
    Epoch 00175: val_accuracy improved from 0.75123 to 0.75616, saving model to ./cnn_v-0.75616.ckpt
    10/10 - 119s - loss: 0.4294 - accuracy: 0.8500 - val_loss: 0.6860 - val_accuracy: 0.7562
    Epoch 176/400
    
    Epoch 00176: val_accuracy did not improve from 0.75616
    10/10 - 119s - loss: 0.4613 - accuracy: 0.8141 - val_loss: 0.7249 - val_accuracy: 0.7328
    Epoch 177/400
    
    Epoch 00177: val_accuracy did not improve from 0.75616
    10/10 - 119s - loss: 0.5268 - accuracy: 0.8109 - val_loss: 0.6557 - val_accuracy: 0.7537
    Epoch 178/400
    
    Epoch 00178: val_accuracy improved from 0.75616 to 0.76478, saving model to ./cnn_v-0.76478.ckpt
    10/10 - 119s - loss: 0.4809 - accuracy: 0.8328 - val_loss: 0.6795 - val_accuracy: 0.7648
    Epoch 179/400
    
    Epoch 00179: val_accuracy did not improve from 0.76478
    10/10 - 119s - loss: 0.3825 - accuracy: 0.8625 - val_loss: 0.7179 - val_accuracy: 0.7537
    Epoch 180/400
    
    Epoch 00180: val_accuracy improved from 0.76478 to 0.76724, saving model to ./cnn_v-0.76724.ckpt
    10/10 - 119s - loss: 0.4182 - accuracy: 0.8531 - val_loss: 0.6953 - val_accuracy: 0.7672
    Epoch 181/400
    
    Epoch 00181: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4994 - accuracy: 0.8281 - val_loss: 0.7276 - val_accuracy: 0.7586
    Epoch 182/400
    
    Epoch 00182: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4839 - accuracy: 0.8156 - val_loss: 0.7101 - val_accuracy: 0.7340
    Epoch 183/400
    
    Epoch 00183: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4360 - accuracy: 0.8484 - val_loss: 0.7482 - val_accuracy: 0.7537
    Epoch 184/400
    
    Epoch 00184: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4660 - accuracy: 0.8313 - val_loss: 0.7222 - val_accuracy: 0.7475
    Epoch 185/400
    
    Epoch 00185: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4645 - accuracy: 0.8406 - val_loss: 0.6569 - val_accuracy: 0.7623
    Epoch 186/400
    
    Epoch 00186: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4422 - accuracy: 0.8422 - val_loss: 0.6764 - val_accuracy: 0.7660
    Epoch 187/400
    
    Epoch 00187: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4168 - accuracy: 0.8516 - val_loss: 0.6647 - val_accuracy: 0.7537
    Epoch 188/400
    
    Epoch 00188: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4558 - accuracy: 0.8547 - val_loss: 0.6733 - val_accuracy: 0.7672
    Epoch 189/400
    
    Epoch 00189: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4690 - accuracy: 0.8250 - val_loss: 0.8590 - val_accuracy: 0.7044
    Epoch 190/400
    
    Epoch 00190: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4641 - accuracy: 0.8344 - val_loss: 0.7261 - val_accuracy: 0.7488
    Epoch 191/400
    
    Epoch 00191: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4456 - accuracy: 0.8375 - val_loss: 0.7740 - val_accuracy: 0.7414
    Epoch 192/400
    
    Epoch 00192: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4955 - accuracy: 0.8203 - val_loss: 0.6855 - val_accuracy: 0.7549
    Epoch 193/400
    
    Epoch 00193: val_accuracy did not improve from 0.76724
    10/10 - 120s - loss: 0.4559 - accuracy: 0.8313 - val_loss: 0.6843 - val_accuracy: 0.7512
    Epoch 194/400
    
    Epoch 00194: val_accuracy did not improve from 0.76724
    10/10 - 119s - loss: 0.4078 - accuracy: 0.8484 - val_loss: 0.6596 - val_accuracy: 0.7586
    Epoch 195/400
    
    Epoch 00195: val_accuracy improved from 0.76724 to 0.77094, saving model to ./cnn_v-0.77094.ckpt
    10/10 - 119s - loss: 0.4786 - accuracy: 0.8469 - val_loss: 0.6535 - val_accuracy: 0.7709
    Epoch 196/400
    
    Epoch 00196: val_accuracy improved from 0.77094 to 0.77586, saving model to ./cnn_v-0.77586.ckpt
    10/10 - 119s - loss: 0.3680 - accuracy: 0.8641 - val_loss: 0.6344 - val_accuracy: 0.7759
    Epoch 197/400
    
    Epoch 00197: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.3588 - accuracy: 0.8687 - val_loss: 0.6874 - val_accuracy: 0.7635
    Epoch 198/400
    
    Epoch 00198: val_accuracy did not improve from 0.77586
    10/10 - 120s - loss: 0.3920 - accuracy: 0.8500 - val_loss: 0.8192 - val_accuracy: 0.7426
    Epoch 199/400
    
    Epoch 00199: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4406 - accuracy: 0.8500 - val_loss: 0.7003 - val_accuracy: 0.7562
    Epoch 200/400
    
    Epoch 00200: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4338 - accuracy: 0.8391 - val_loss: 0.6831 - val_accuracy: 0.7635
    Epoch 201/400
    
    Epoch 00201: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.3889 - accuracy: 0.8672 - val_loss: 0.6445 - val_accuracy: 0.7685
    Epoch 202/400
    
    Epoch 00202: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4226 - accuracy: 0.8484 - val_loss: 0.6521 - val_accuracy: 0.7685
    Epoch 203/400
    
    Epoch 00203: val_accuracy did not improve from 0.77586
    10/10 - 120s - loss: 0.3699 - accuracy: 0.8672 - val_loss: 0.7135 - val_accuracy: 0.7623
    Epoch 204/400
    
    Epoch 00204: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4217 - accuracy: 0.8422 - val_loss: 0.7046 - val_accuracy: 0.7611
    Epoch 205/400
    
    Epoch 00205: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.3705 - accuracy: 0.8656 - val_loss: 0.6606 - val_accuracy: 0.7623
    Epoch 206/400
    
    Epoch 00206: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.3733 - accuracy: 0.8734 - val_loss: 0.6492 - val_accuracy: 0.7685
    Epoch 207/400
    
    Epoch 00207: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4120 - accuracy: 0.8656 - val_loss: 0.7590 - val_accuracy: 0.7414
    Epoch 208/400
    
    Epoch 00208: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.5093 - accuracy: 0.8234 - val_loss: 0.7555 - val_accuracy: 0.7340
    Epoch 209/400
    
    Epoch 00209: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4704 - accuracy: 0.8313 - val_loss: 0.6836 - val_accuracy: 0.7562
    Epoch 210/400
    
    Epoch 00210: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4072 - accuracy: 0.8625 - val_loss: 0.6694 - val_accuracy: 0.7537
    Epoch 211/400
    
    Epoch 00211: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4354 - accuracy: 0.8500 - val_loss: 0.7070 - val_accuracy: 0.7611
    Epoch 212/400
    
    Epoch 00212: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.4354 - accuracy: 0.8516 - val_loss: 0.7333 - val_accuracy: 0.7303
    Epoch 213/400
    
    Epoch 00213: val_accuracy did not improve from 0.77586
    10/10 - 119s - loss: 0.3933 - accuracy: 0.8609 - val_loss: 0.6658 - val_accuracy: 0.7623
    Epoch 214/400
    
    Epoch 00214: val_accuracy improved from 0.77586 to 0.78695, saving model to ./cnn_v-0.78695.ckpt
    10/10 - 119s - loss: 0.3528 - accuracy: 0.8578 - val_loss: 0.6373 - val_accuracy: 0.7869
    Epoch 215/400
    
    Epoch 00215: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3086 - accuracy: 0.8859 - val_loss: 0.6703 - val_accuracy: 0.7771
    Epoch 216/400
    
    Epoch 00216: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3597 - accuracy: 0.8891 - val_loss: 0.7076 - val_accuracy: 0.7414
    Epoch 217/400
    
    Epoch 00217: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3912 - accuracy: 0.8469 - val_loss: 0.6902 - val_accuracy: 0.7623
    Epoch 218/400
    
    Epoch 00218: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3877 - accuracy: 0.8594 - val_loss: 0.7294 - val_accuracy: 0.7586
    Epoch 219/400
    
    Epoch 00219: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4421 - accuracy: 0.8391 - val_loss: 0.6990 - val_accuracy: 0.7549
    Epoch 220/400
    
    Epoch 00220: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4075 - accuracy: 0.8469 - val_loss: 0.7694 - val_accuracy: 0.7537
    Epoch 221/400
    
    Epoch 00221: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4079 - accuracy: 0.8578 - val_loss: 0.7153 - val_accuracy: 0.7672
    Epoch 222/400
    
    Epoch 00222: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3906 - accuracy: 0.8687 - val_loss: 0.7459 - val_accuracy: 0.7512
    Epoch 223/400
    
    Epoch 00223: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4299 - accuracy: 0.8578 - val_loss: 0.7149 - val_accuracy: 0.7525
    Epoch 224/400
    
    Epoch 00224: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3428 - accuracy: 0.8828 - val_loss: 0.7002 - val_accuracy: 0.7537
    Epoch 225/400
    
    Epoch 00225: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.3457 - accuracy: 0.8844 - val_loss: 0.6647 - val_accuracy: 0.7796
    Epoch 226/400
    
    Epoch 00226: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4129 - accuracy: 0.8594 - val_loss: 0.7327 - val_accuracy: 0.7475
    Epoch 227/400
    
    Epoch 00227: val_accuracy did not improve from 0.78695
    10/10 - 119s - loss: 0.4061 - accuracy: 0.8703 - val_loss: 0.6699 - val_accuracy: 0.7746
    Epoch 228/400
    
    Epoch 00228: val_accuracy improved from 0.78695 to 0.79187, saving model to ./cnn_v-0.79187.ckpt
    10/10 - 119s - loss: 0.3510 - accuracy: 0.8734 - val_loss: 0.6609 - val_accuracy: 0.7919
    Epoch 229/400
    
    Epoch 00229: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3435 - accuracy: 0.8844 - val_loss: 0.7130 - val_accuracy: 0.7796
    Epoch 230/400
    
    Epoch 00230: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3729 - accuracy: 0.8562 - val_loss: 0.6415 - val_accuracy: 0.7857
    Epoch 231/400
    
    Epoch 00231: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3337 - accuracy: 0.8953 - val_loss: 0.7036 - val_accuracy: 0.7783
    Epoch 232/400
    
    Epoch 00232: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3249 - accuracy: 0.8906 - val_loss: 0.6854 - val_accuracy: 0.7709
    Epoch 233/400
    
    Epoch 00233: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3340 - accuracy: 0.8969 - val_loss: 0.7489 - val_accuracy: 0.7438
    Epoch 234/400
    
    Epoch 00234: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.3397 - accuracy: 0.8656 - val_loss: 0.6727 - val_accuracy: 0.7660
    Epoch 235/400
    
    Epoch 00235: val_accuracy did not improve from 0.79187
    10/10 - 119s - loss: 0.2902 - accuracy: 0.9000 - val_loss: 0.6765 - val_accuracy: 0.7845
    Epoch 236/400
    
    Epoch 00236: val_accuracy improved from 0.79187 to 0.79803, saving model to ./cnn_v-0.79803.ckpt
    10/10 - 119s - loss: 0.3006 - accuracy: 0.8984 - val_loss: 0.6474 - val_accuracy: 0.7980
    Epoch 237/400
    
    Epoch 00237: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3432 - accuracy: 0.8766 - val_loss: 0.6767 - val_accuracy: 0.7759
    Epoch 238/400
    
    Epoch 00238: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3332 - accuracy: 0.8813 - val_loss: 0.7027 - val_accuracy: 0.7746
    Epoch 239/400
    
    Epoch 00239: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.3019 - accuracy: 0.8969 - val_loss: 0.6452 - val_accuracy: 0.7796
    Epoch 240/400
    
    Epoch 00240: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3340 - accuracy: 0.8859 - val_loss: 0.6943 - val_accuracy: 0.7894
    Epoch 241/400
    
    Epoch 00241: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3938 - accuracy: 0.8625 - val_loss: 0.6764 - val_accuracy: 0.7697
    Epoch 242/400
    
    Epoch 00242: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3275 - accuracy: 0.8859 - val_loss: 0.7860 - val_accuracy: 0.7525
    Epoch 243/400
    
    Epoch 00243: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3478 - accuracy: 0.8766 - val_loss: 0.6596 - val_accuracy: 0.7734
    Epoch 244/400
    
    Epoch 00244: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.3015 - accuracy: 0.9031 - val_loss: 0.7339 - val_accuracy: 0.7648
    Epoch 245/400
    
    Epoch 00245: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3478 - accuracy: 0.8781 - val_loss: 0.6911 - val_accuracy: 0.7845
    Epoch 246/400
    
    Epoch 00246: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3386 - accuracy: 0.8844 - val_loss: 0.6886 - val_accuracy: 0.7722
    Epoch 247/400
    
    Epoch 00247: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2594 - accuracy: 0.9281 - val_loss: 0.6893 - val_accuracy: 0.7820
    Epoch 248/400
    
    Epoch 00248: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2786 - accuracy: 0.9031 - val_loss: 0.7958 - val_accuracy: 0.7525
    Epoch 249/400
    
    Epoch 00249: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.3492 - accuracy: 0.8734 - val_loss: 0.8105 - val_accuracy: 0.7340
    Epoch 250/400
    
    Epoch 00250: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3551 - accuracy: 0.8672 - val_loss: 0.7543 - val_accuracy: 0.7475
    Epoch 251/400
    
    Epoch 00251: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3071 - accuracy: 0.9031 - val_loss: 0.7001 - val_accuracy: 0.7697
    Epoch 252/400
    
    Epoch 00252: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3002 - accuracy: 0.8969 - val_loss: 0.6881 - val_accuracy: 0.7759
    Epoch 253/400
    
    Epoch 00253: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2996 - accuracy: 0.8922 - val_loss: 0.7206 - val_accuracy: 0.7709
    Epoch 254/400
    
    Epoch 00254: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2628 - accuracy: 0.9109 - val_loss: 0.6634 - val_accuracy: 0.7956
    Epoch 255/400
    
    Epoch 00255: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2226 - accuracy: 0.9312 - val_loss: 0.6848 - val_accuracy: 0.7956
    Epoch 256/400
    
    Epoch 00256: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2755 - accuracy: 0.8891 - val_loss: 0.7465 - val_accuracy: 0.7759
    Epoch 257/400
    
    Epoch 00257: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3039 - accuracy: 0.8844 - val_loss: 0.7359 - val_accuracy: 0.7759
    Epoch 258/400
    
    Epoch 00258: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2482 - accuracy: 0.9078 - val_loss: 0.8100 - val_accuracy: 0.7746
    Epoch 259/400
    
    Epoch 00259: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2438 - accuracy: 0.9187 - val_loss: 0.7432 - val_accuracy: 0.7722
    Epoch 260/400
    
    Epoch 00260: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2902 - accuracy: 0.8906 - val_loss: 0.8452 - val_accuracy: 0.7599
    Epoch 261/400
    
    Epoch 00261: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.4100 - accuracy: 0.8641 - val_loss: 0.7755 - val_accuracy: 0.7709
    Epoch 262/400
    
    Epoch 00262: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2613 - accuracy: 0.9078 - val_loss: 0.6790 - val_accuracy: 0.7833
    Epoch 263/400
    
    Epoch 00263: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3149 - accuracy: 0.8844 - val_loss: 0.7507 - val_accuracy: 0.7709
    Epoch 264/400
    
    Epoch 00264: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2948 - accuracy: 0.8906 - val_loss: 0.7048 - val_accuracy: 0.7771
    Epoch 265/400
    
    Epoch 00265: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2602 - accuracy: 0.9062 - val_loss: 0.7039 - val_accuracy: 0.7808
    Epoch 266/400
    
    Epoch 00266: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2347 - accuracy: 0.9187 - val_loss: 0.7364 - val_accuracy: 0.7919
    Epoch 267/400
    
    Epoch 00267: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2333 - accuracy: 0.9109 - val_loss: 0.7150 - val_accuracy: 0.7759
    Epoch 268/400
    
    Epoch 00268: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2478 - accuracy: 0.9125 - val_loss: 0.7554 - val_accuracy: 0.7722
    Epoch 269/400
    
    Epoch 00269: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2791 - accuracy: 0.9125 - val_loss: 0.8053 - val_accuracy: 0.7586
    Epoch 270/400
    
    Epoch 00270: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2731 - accuracy: 0.9094 - val_loss: 0.7755 - val_accuracy: 0.7635
    Epoch 271/400
    
    Epoch 00271: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2777 - accuracy: 0.9016 - val_loss: 0.8934 - val_accuracy: 0.7451
    Epoch 272/400
    
    Epoch 00272: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3201 - accuracy: 0.8828 - val_loss: 0.7635 - val_accuracy: 0.7648
    Epoch 273/400
    
    Epoch 00273: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2451 - accuracy: 0.9187 - val_loss: 0.7544 - val_accuracy: 0.7660
    Epoch 274/400
    
    Epoch 00274: val_accuracy did not improve from 0.79803
    10/10 - 121s - loss: 0.2939 - accuracy: 0.8938 - val_loss: 0.8936 - val_accuracy: 0.7069
    Epoch 275/400
    
    Epoch 00275: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.3245 - accuracy: 0.8906 - val_loss: 0.6646 - val_accuracy: 0.7771
    Epoch 276/400
    
    Epoch 00276: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2248 - accuracy: 0.9125 - val_loss: 0.8057 - val_accuracy: 0.7722
    Epoch 277/400
    
    Epoch 00277: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3021 - accuracy: 0.9000 - val_loss: 0.6808 - val_accuracy: 0.7808
    Epoch 278/400
    
    Epoch 00278: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2729 - accuracy: 0.9031 - val_loss: 0.6717 - val_accuracy: 0.7783
    Epoch 279/400
    
    Epoch 00279: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2137 - accuracy: 0.9297 - val_loss: 0.7484 - val_accuracy: 0.7709
    Epoch 280/400
    
    Epoch 00280: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2344 - accuracy: 0.9094 - val_loss: 0.7376 - val_accuracy: 0.7980
    Epoch 281/400
    
    Epoch 00281: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2493 - accuracy: 0.9141 - val_loss: 0.8927 - val_accuracy: 0.7648
    Epoch 282/400
    
    Epoch 00282: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2397 - accuracy: 0.9156 - val_loss: 0.7235 - val_accuracy: 0.7820
    Epoch 283/400
    
    Epoch 00283: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2300 - accuracy: 0.9266 - val_loss: 0.7794 - val_accuracy: 0.7709
    Epoch 284/400
    
    Epoch 00284: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2738 - accuracy: 0.9016 - val_loss: 0.8098 - val_accuracy: 0.7746
    Epoch 285/400
    
    Epoch 00285: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2599 - accuracy: 0.9000 - val_loss: 0.7544 - val_accuracy: 0.7783
    Epoch 286/400
    
    Epoch 00286: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2122 - accuracy: 0.9250 - val_loss: 0.7513 - val_accuracy: 0.7869
    Epoch 287/400
    
    Epoch 00287: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2248 - accuracy: 0.9250 - val_loss: 0.6731 - val_accuracy: 0.7894
    Epoch 288/400
    
    Epoch 00288: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1998 - accuracy: 0.9250 - val_loss: 0.7645 - val_accuracy: 0.7869
    Epoch 289/400
    
    Epoch 00289: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1551 - accuracy: 0.9500 - val_loss: 0.8397 - val_accuracy: 0.7833
    Epoch 290/400
    
    Epoch 00290: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2297 - accuracy: 0.9156 - val_loss: 0.7943 - val_accuracy: 0.7672
    Epoch 291/400
    
    Epoch 00291: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2053 - accuracy: 0.9203 - val_loss: 0.7711 - val_accuracy: 0.7746
    Epoch 292/400
    
    Epoch 00292: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2262 - accuracy: 0.9219 - val_loss: 0.7748 - val_accuracy: 0.7660
    Epoch 293/400
    
    Epoch 00293: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2097 - accuracy: 0.9328 - val_loss: 0.7773 - val_accuracy: 0.7820
    Epoch 294/400
    
    Epoch 00294: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1634 - accuracy: 0.9312 - val_loss: 0.8015 - val_accuracy: 0.7660
    Epoch 295/400
    
    Epoch 00295: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1804 - accuracy: 0.9297 - val_loss: 0.8263 - val_accuracy: 0.7833
    Epoch 296/400
    
    Epoch 00296: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2131 - accuracy: 0.9250 - val_loss: 0.8233 - val_accuracy: 0.7648
    Epoch 297/400
    
    Epoch 00297: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1906 - accuracy: 0.9453 - val_loss: 0.7927 - val_accuracy: 0.7796
    Epoch 298/400
    
    Epoch 00298: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2121 - accuracy: 0.9297 - val_loss: 0.7863 - val_accuracy: 0.7709
    Epoch 299/400
    
    Epoch 00299: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1952 - accuracy: 0.9266 - val_loss: 0.8549 - val_accuracy: 0.7623
    Epoch 300/400
    
    Epoch 00300: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2023 - accuracy: 0.9297 - val_loss: 0.8188 - val_accuracy: 0.7820
    Epoch 301/400
    
    Epoch 00301: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.1714 - accuracy: 0.9453 - val_loss: 0.8343 - val_accuracy: 0.7869
    Epoch 302/400
    
    Epoch 00302: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1583 - accuracy: 0.9469 - val_loss: 0.8659 - val_accuracy: 0.7734
    Epoch 303/400
    
    Epoch 00303: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2190 - accuracy: 0.9219 - val_loss: 0.8764 - val_accuracy: 0.7759
    Epoch 304/400
    
    Epoch 00304: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1427 - accuracy: 0.9453 - val_loss: 0.7476 - val_accuracy: 0.7882
    Epoch 305/400
    
    Epoch 00305: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2121 - accuracy: 0.9234 - val_loss: 0.8855 - val_accuracy: 0.7685
    Epoch 306/400
    
    Epoch 00306: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2157 - accuracy: 0.9219 - val_loss: 0.7741 - val_accuracy: 0.7956
    Epoch 307/400
    
    Epoch 00307: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1819 - accuracy: 0.9344 - val_loss: 0.9035 - val_accuracy: 0.7709
    Epoch 308/400
    
    Epoch 00308: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1766 - accuracy: 0.9422 - val_loss: 0.8693 - val_accuracy: 0.7722
    Epoch 309/400
    
    Epoch 00309: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2281 - accuracy: 0.9219 - val_loss: 1.0119 - val_accuracy: 0.7303
    Epoch 310/400
    
    Epoch 00310: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.3022 - accuracy: 0.8938 - val_loss: 0.6939 - val_accuracy: 0.7931
    Epoch 311/400
    
    Epoch 00311: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2642 - accuracy: 0.9094 - val_loss: 0.8591 - val_accuracy: 0.7623
    Epoch 312/400
    
    Epoch 00312: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1719 - accuracy: 0.9344 - val_loss: 0.7859 - val_accuracy: 0.7759
    Epoch 313/400
    
    Epoch 00313: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1971 - accuracy: 0.9250 - val_loss: 0.8726 - val_accuracy: 0.7771
    Epoch 314/400
    
    Epoch 00314: val_accuracy did not improve from 0.79803
    10/10 - 121s - loss: 0.2222 - accuracy: 0.9281 - val_loss: 0.7722 - val_accuracy: 0.7833
    Epoch 315/400
    
    Epoch 00315: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2332 - accuracy: 0.9234 - val_loss: 0.8147 - val_accuracy: 0.7746
    Epoch 316/400
    
    Epoch 00316: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1871 - accuracy: 0.9250 - val_loss: 0.7771 - val_accuracy: 0.7796
    Epoch 317/400
    
    Epoch 00317: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1863 - accuracy: 0.9453 - val_loss: 0.7720 - val_accuracy: 0.7611
    Epoch 318/400
    
    Epoch 00318: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2073 - accuracy: 0.9187 - val_loss: 0.8009 - val_accuracy: 0.7906
    Epoch 319/400
    
    Epoch 00319: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1752 - accuracy: 0.9406 - val_loss: 0.7575 - val_accuracy: 0.7845
    Epoch 320/400
    
    Epoch 00320: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1460 - accuracy: 0.9594 - val_loss: 0.7722 - val_accuracy: 0.7869
    Epoch 321/400
    
    Epoch 00321: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.1034 - accuracy: 0.9719 - val_loss: 0.9066 - val_accuracy: 0.7722
    Epoch 322/400
    
    Epoch 00322: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1934 - accuracy: 0.9328 - val_loss: 0.8480 - val_accuracy: 0.7672
    Epoch 323/400
    
    Epoch 00323: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1859 - accuracy: 0.9375 - val_loss: 0.7599 - val_accuracy: 0.7808
    Epoch 324/400
    
    Epoch 00324: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1456 - accuracy: 0.9422 - val_loss: 0.7841 - val_accuracy: 0.7919
    Epoch 325/400
    
    Epoch 00325: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2123 - accuracy: 0.9234 - val_loss: 0.8564 - val_accuracy: 0.7660
    Epoch 326/400
    
    Epoch 00326: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1186 - accuracy: 0.9578 - val_loss: 0.8363 - val_accuracy: 0.7808
    Epoch 327/400
    
    Epoch 00327: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1219 - accuracy: 0.9609 - val_loss: 0.8208 - val_accuracy: 0.7882
    Epoch 328/400
    
    Epoch 00328: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1613 - accuracy: 0.9531 - val_loss: 0.8735 - val_accuracy: 0.7734
    Epoch 329/400
    
    Epoch 00329: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2012 - accuracy: 0.9281 - val_loss: 0.7859 - val_accuracy: 0.7820
    Epoch 330/400
    
    Epoch 00330: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2380 - accuracy: 0.9156 - val_loss: 0.9362 - val_accuracy: 0.7525
    Epoch 331/400
    
    Epoch 00331: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1558 - accuracy: 0.9391 - val_loss: 0.9659 - val_accuracy: 0.7660
    Epoch 332/400
    
    Epoch 00332: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1937 - accuracy: 0.9250 - val_loss: 0.8037 - val_accuracy: 0.7833
    Epoch 333/400
    
    Epoch 00333: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2187 - accuracy: 0.9234 - val_loss: 0.8438 - val_accuracy: 0.7820
    Epoch 334/400
    
    Epoch 00334: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2172 - accuracy: 0.9234 - val_loss: 0.8451 - val_accuracy: 0.7586
    Epoch 335/400
    
    Epoch 00335: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1821 - accuracy: 0.9328 - val_loss: 0.8523 - val_accuracy: 0.7857
    Epoch 336/400
    
    Epoch 00336: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1816 - accuracy: 0.9453 - val_loss: 1.0622 - val_accuracy: 0.7500
    Epoch 337/400
    
    Epoch 00337: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2183 - accuracy: 0.9234 - val_loss: 0.8907 - val_accuracy: 0.7451
    Epoch 338/400
    
    Epoch 00338: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2281 - accuracy: 0.9125 - val_loss: 0.9147 - val_accuracy: 0.7660
    Epoch 339/400
    
    Epoch 00339: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1843 - accuracy: 0.9359 - val_loss: 0.8315 - val_accuracy: 0.7869
    Epoch 340/400
    
    Epoch 00340: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1834 - accuracy: 0.9328 - val_loss: 0.9109 - val_accuracy: 0.7611
    Epoch 341/400
    
    Epoch 00341: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1372 - accuracy: 0.9531 - val_loss: 0.8330 - val_accuracy: 0.7697
    Epoch 342/400
    
    Epoch 00342: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1494 - accuracy: 0.9531 - val_loss: 1.0399 - val_accuracy: 0.7500
    Epoch 343/400
    
    Epoch 00343: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1673 - accuracy: 0.9469 - val_loss: 0.8525 - val_accuracy: 0.7833
    Epoch 344/400
    
    Epoch 00344: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1403 - accuracy: 0.9469 - val_loss: 0.8694 - val_accuracy: 0.7833
    Epoch 345/400
    
    Epoch 00345: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1121 - accuracy: 0.9672 - val_loss: 0.9483 - val_accuracy: 0.7599
    Epoch 346/400
    
    Epoch 00346: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1238 - accuracy: 0.9594 - val_loss: 0.8713 - val_accuracy: 0.7869
    Epoch 347/400
    
    Epoch 00347: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0931 - accuracy: 0.9672 - val_loss: 1.0321 - val_accuracy: 0.7599
    Epoch 348/400
    
    Epoch 00348: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1190 - accuracy: 0.9656 - val_loss: 0.9625 - val_accuracy: 0.7734
    Epoch 349/400
    
    Epoch 00349: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1468 - accuracy: 0.9500 - val_loss: 0.8977 - val_accuracy: 0.7845
    Epoch 350/400
    
    Epoch 00350: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1377 - accuracy: 0.9594 - val_loss: 1.0045 - val_accuracy: 0.7759
    Epoch 351/400
    
    Epoch 00351: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1315 - accuracy: 0.9484 - val_loss: 0.8783 - val_accuracy: 0.7820
    Epoch 352/400
    
    Epoch 00352: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1244 - accuracy: 0.9625 - val_loss: 0.9731 - val_accuracy: 0.7586
    Epoch 353/400
    
    Epoch 00353: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1575 - accuracy: 0.9359 - val_loss: 0.8728 - val_accuracy: 0.7943
    Epoch 354/400
    
    Epoch 00354: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0911 - accuracy: 0.9750 - val_loss: 0.8557 - val_accuracy: 0.7857
    Epoch 355/400
    
    Epoch 00355: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.1387 - accuracy: 0.9609 - val_loss: 0.8893 - val_accuracy: 0.7906
    Epoch 356/400
    
    Epoch 00356: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1081 - accuracy: 0.9594 - val_loss: 0.9322 - val_accuracy: 0.7697
    Epoch 357/400
    
    Epoch 00357: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0913 - accuracy: 0.9719 - val_loss: 0.9715 - val_accuracy: 0.7759
    Epoch 358/400
    
    Epoch 00358: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0923 - accuracy: 0.9750 - val_loss: 0.9890 - val_accuracy: 0.7648
    Epoch 359/400
    
    Epoch 00359: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0950 - accuracy: 0.9609 - val_loss: 1.0334 - val_accuracy: 0.7709
    Epoch 360/400
    
    Epoch 00360: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0833 - accuracy: 0.9703 - val_loss: 0.9636 - val_accuracy: 0.7746
    Epoch 361/400
    
    Epoch 00361: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1058 - accuracy: 0.9641 - val_loss: 1.0360 - val_accuracy: 0.7611
    Epoch 362/400
    
    Epoch 00362: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0810 - accuracy: 0.9781 - val_loss: 1.0426 - val_accuracy: 0.7869
    Epoch 363/400
    
    Epoch 00363: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.1143 - accuracy: 0.9516 - val_loss: 1.0522 - val_accuracy: 0.7685
    Epoch 364/400
    
    Epoch 00364: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0901 - accuracy: 0.9578 - val_loss: 1.0033 - val_accuracy: 0.7759
    Epoch 365/400
    
    Epoch 00365: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1006 - accuracy: 0.9656 - val_loss: 1.0868 - val_accuracy: 0.7635
    Epoch 366/400
    
    Epoch 00366: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2045 - accuracy: 0.9312 - val_loss: 1.0962 - val_accuracy: 0.7549
    Epoch 367/400
    
    Epoch 00367: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1624 - accuracy: 0.9391 - val_loss: 1.0585 - val_accuracy: 0.7549
    Epoch 368/400
    
    Epoch 00368: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2097 - accuracy: 0.9312 - val_loss: 0.9780 - val_accuracy: 0.7672
    Epoch 369/400
    
    Epoch 00369: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1736 - accuracy: 0.9375 - val_loss: 1.0019 - val_accuracy: 0.7635
    Epoch 370/400
    
    Epoch 00370: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1351 - accuracy: 0.9484 - val_loss: 0.9144 - val_accuracy: 0.7943
    Epoch 371/400
    
    Epoch 00371: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1124 - accuracy: 0.9578 - val_loss: 1.0294 - val_accuracy: 0.7709
    Epoch 372/400
    
    Epoch 00372: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1090 - accuracy: 0.9609 - val_loss: 1.0078 - val_accuracy: 0.7845
    Epoch 373/400
    
    Epoch 00373: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.1016 - accuracy: 0.9750 - val_loss: 0.9745 - val_accuracy: 0.7796
    Epoch 374/400
    
    Epoch 00374: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1183 - accuracy: 0.9516 - val_loss: 1.0422 - val_accuracy: 0.7833
    Epoch 375/400
    
    Epoch 00375: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1768 - accuracy: 0.9312 - val_loss: 0.9147 - val_accuracy: 0.7869
    Epoch 376/400
    
    Epoch 00376: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1315 - accuracy: 0.9547 - val_loss: 0.9429 - val_accuracy: 0.7734
    Epoch 377/400
    
    Epoch 00377: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0887 - accuracy: 0.9750 - val_loss: 0.9299 - val_accuracy: 0.7919
    Epoch 378/400
    
    Epoch 00378: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1283 - accuracy: 0.9625 - val_loss: 0.9058 - val_accuracy: 0.7833
    Epoch 379/400
    
    Epoch 00379: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1050 - accuracy: 0.9578 - val_loss: 0.9781 - val_accuracy: 0.7869
    Epoch 380/400
    
    Epoch 00380: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0671 - accuracy: 0.9828 - val_loss: 1.1961 - val_accuracy: 0.7783
    Epoch 381/400
    
    Epoch 00381: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0657 - accuracy: 0.9812 - val_loss: 0.9827 - val_accuracy: 0.7771
    Epoch 382/400
    
    Epoch 00382: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0688 - accuracy: 0.9734 - val_loss: 1.0376 - val_accuracy: 0.7771
    Epoch 383/400
    
    Epoch 00383: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1019 - accuracy: 0.9703 - val_loss: 1.0070 - val_accuracy: 0.7796
    Epoch 384/400
    
    Epoch 00384: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1134 - accuracy: 0.9563 - val_loss: 1.1213 - val_accuracy: 0.7623
    Epoch 385/400
    
    Epoch 00385: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0909 - accuracy: 0.9703 - val_loss: 1.1585 - val_accuracy: 0.7722
    Epoch 386/400
    
    Epoch 00386: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0957 - accuracy: 0.9578 - val_loss: 1.1287 - val_accuracy: 0.7500
    Epoch 387/400
    
    Epoch 00387: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1092 - accuracy: 0.9547 - val_loss: 1.0201 - val_accuracy: 0.7869
    Epoch 388/400
    
    Epoch 00388: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.0768 - accuracy: 0.9828 - val_loss: 0.9607 - val_accuracy: 0.7808
    Epoch 389/400
    
    Epoch 00389: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0719 - accuracy: 0.9781 - val_loss: 1.0544 - val_accuracy: 0.7820
    Epoch 390/400
    
    Epoch 00390: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0678 - accuracy: 0.9766 - val_loss: 1.0490 - val_accuracy: 0.7833
    Epoch 391/400
    
    Epoch 00391: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0626 - accuracy: 0.9828 - val_loss: 1.3636 - val_accuracy: 0.7315
    Epoch 392/400
    
    Epoch 00392: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1053 - accuracy: 0.9516 - val_loss: 1.1999 - val_accuracy: 0.7771
    Epoch 393/400
    
    Epoch 00393: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.0859 - accuracy: 0.9734 - val_loss: 1.2883 - val_accuracy: 0.7475
    Epoch 394/400
    
    Epoch 00394: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1067 - accuracy: 0.9609 - val_loss: 1.0585 - val_accuracy: 0.7697
    Epoch 395/400
    
    Epoch 00395: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1202 - accuracy: 0.9594 - val_loss: 1.2605 - val_accuracy: 0.7438
    Epoch 396/400
    
    Epoch 00396: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1353 - accuracy: 0.9438 - val_loss: 1.1009 - val_accuracy: 0.7512
    Epoch 397/400
    
    Epoch 00397: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2092 - accuracy: 0.9328 - val_loss: 1.1955 - val_accuracy: 0.7266
    Epoch 398/400
    
    Epoch 00398: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.2369 - accuracy: 0.9203 - val_loss: 1.0231 - val_accuracy: 0.7722
    Epoch 399/400
    
    Epoch 00399: val_accuracy did not improve from 0.79803
    10/10 - 120s - loss: 0.2069 - accuracy: 0.9109 - val_loss: 1.1256 - val_accuracy: 0.7389
    Epoch 400/400
    
    Epoch 00400: val_accuracy did not improve from 0.79803
    10/10 - 119s - loss: 0.1796 - accuracy: 0.9250 - val_loss: 0.9489 - val_accuracy: 0.7845
    

## 5、CNN可视化

加载最佳模型 


```python
# 加载已经训练好的参数
model_best= build_cnn()
best_para_weight = tf.train.latest_checkpoint(checkpoint_dir_base)
model_best.load_weights(best_para_weight)
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x56fe948>



+ ### 5.1 可视化模型每层的输出

在测试数据中随机选择第一个图片，查看模型中的每一层针对此图片的输出。


```python
# 随机打乱
TEST_DATA = TEST_DATA.shuffle(400)
# 选择第一个
for d in TEST_DATA.take(1):
    x, label = d
    # 图片数据
    fig_data = x.numpy()
    # 图片对应的标签数据
    label_data = class_list[label.numpy()]
    # 添加维度
    add_fig_data = fig_data[None,:,:,:]
    # 模型预测结果
    result = model_best.predict(add_fig_data)
    # 模型预测类别
    predict_lable = class_list[np.argmax(result, axis=1)[0]]

    # 每一行图片的个数
    column = 12
    # 查看每一层的输出
    layer = model_best.layers  # 所有层
    for l in layer:  # 遍历层
        # 输出
        add_fig_data = l(add_fig_data)  
        # 绘制层的输出
        count = add_fig_data.shape[-1]
        # 去掉添加的维度
        s_data = tf.squeeze(add_fig_data)
        # 需要的行数
        row = ceil(count / column)
        if len(s_data.shape) == 3:  
            filer_data = tf.transpose(s_data, perm=[2, 0, 1])
            fig = plt.figure(figsize=(10, 5))
            sign = 1
            if filer_data.shape[0] == 3: # 原始图片，
                fig.add_subplot(1,1,sign)
                plt.axis('off')
                plt.imshow(s_data)
            else:
                for sd in filer_data:
                    fig.add_subplot(row,column,sign)
                    plt.axis('off')
                    plt.imshow(sd)
                    sign += 1
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.suptitle('{}层 输出'.format(l.name))
            plt.show()
    # 输出预测结果的各个类别的柱状图以及
    plt.figure()
    plt.bar(class_list, result[0])
    plt.title('输出层结果柱状图(该图片真实类别：{})'.format(label_data))
    plt.xlabel('类别')
    plt.ylabel('概率')
    plt.show()
        
```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)



![png](output_30_3.png)



![png](output_30_4.png)



![png](output_30_5.png)



![png](output_30_6.png)



![png](output_30_7.png)



![png](output_30_8.png)



![png](output_30_9.png)



![png](output_30_10.png)


+ ### 5.2 可视化每个卷积层学到的特征

可视化卷积层中的每个卷积核学到的特征，并不是待模型训练完毕后，将该卷积核展示出来，因为卷积核的高度和宽度一般都很小，直接展示出来作用不大，并不能清晰的看出学到的特征。一般有2种方式，一是利用反卷积，实现比较困难；二是下面利用梯度上升的思想来做，通过合成一个图片，使得该卷积核获得最大的激活值，经过一定次数的优化，最终这个图片就可认为是该卷积核学到的特征，步骤如下：
+ 给定训练好的模型一个随机噪声的初始图，将这张图作为模型的输入$x$，计算其在模型中第$i$层$j$个卷积核的激活值$A_{ij}(x)$，也就是输出值；
+ 计算梯度$$\frac{\delta A_{ij}(x)}{\delta x}$$做一个反向传播，也就是用该图的卷积核梯度来更新噪声图：$$x  += \eta \frac{\delta A_{ij}(x)}{\delta x}$$其中$\eta$为学习率；

以此通过改变每个像素的颜色值以增加对该卷积核的激活值。


将浮点类型的数据变为像素值，并平滑，便于图像的展示


```python
# 将浮点值转换成像素值
def deprocess_image(x):
    # 对张量进行规范化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # 转化到RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```

 根据卷积层的层索引输出该层某卷积核学习到的特征


```python
# 根据卷积层的层索引输出该层某卷积核学习到的特征
def get_filter(model, convsign, filtersign, initfig, times, lr):
    """
    model:训练好的模型
    convsign:模型中某个卷积层的索引
    filtersign: 卷积层的卷积核索引，不得大于该卷积层卷积核的个数
    initfig:初始化图片
    times:迭代的次数
    lr:类似学习率
    """
    # 开始迭代
    for i in range(1, times+1):
        # tensorflow2计算梯度的方式
        with tf.GradientTape(persistent=True) as gtape:
            # 加入变量
            gtape.watch(initfig)
            # 输入层
            middle_data = model.layers[0](initfig)
            # 正向传播
            for l in range(1, convsign+1):
                middle_data = model.layers[l](middle_data)
            # 获得该卷积核的损失
            loss = tf.math.reduce_mean(middle_data[:, :, :, filtersign])
        # 计算梯度
        grads  = gtape.gradient(loss, initfig)
        # 平滑梯度
        grads /= (tf.math.sqrt(tf.math.reduce_mean(tf.math.square(grads))) + 1e-5)
        # 更改图片
        initfig += grads * lr
    # 浮点类型转化为像素值
    return deprocess_image(initfig.numpy()[0])
```

输出某模型所有卷积层中所有卷积核学习到的特征


```python
# 首先获取模型中所有卷积层：卷积层索引：[卷积层名称，卷积核个数]
def get_conv(model):
    layer_dict = dict([(sign, [layer.name, len(layer.get_weights()[1])]) 
                   for sign, layer in enumerate(model_best.layers) 
                   if isinstance(layer, tf.keras.layers.Conv2D)])
    return layer_dict
```


```python
# 展示模型中所有的卷积层学到的特征
def show_conv(model, times=60, lr=0.7, column=12):
    layer_dict = get_conv(model)
    # 初始图片
    plt.figure(figsize=(5, 5))
    # 随机噪声数据
    fig_start = tf.constant(np.random.random((1, 192, 192, 3)))  # 维度需要和模型的一致
    plt.imshow(deprocess_image(fig_start.numpy()[0]))
    plt.xticks([])
    plt.yticks([])
    plt.title('初始图片')
    plt.show()

    # 遍历卷积层
    for c in layer_dict:
        # 图片
        fig = plt.figure(figsize=(12, 5))
        # 卷积核个数
        count = layer_dict[c][1]
        row =ceil(count/ column)
        for f in range(count):
            data = get_filter(model, c, f, fig_start, times, lr)
            fig.add_subplot(row, column, f+1)
            plt.axis('off')
            plt.imshow(data)
        plt.suptitle('卷积层：%s学到的特征' % layer_dict[c][0])
        plt.show()
        
show_conv(model_best)    
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)



![png](output_38_4.png)



![png](output_38_5.png)



![png](output_38_6.png)



![png](output_38_7.png)


+ ### 5.3 类激活图可视化

  + **（1）类激活图（CAM， class activation map）**

类激活热力图是针对训练好的模型，输入属于某类的一个图片，输出的是这张图片中各个位置对于这个类的贡献程度的热力图。

实现CAM对模型结构是有要求的，也就是模型的倒数第二层必须是GAP层(GlobalAveragePooling，全局均值池化),并且该层的输出必须等于类别数，因为池化层的输出是根据前面的卷积层的卷积核的个数来的，所以这就要求前面的卷积核的个数为类别数。

下面以本文中的例子说一下如何实现CAM，首先看下面的图片：
 

上面图示模型中最后一个卷积层中浓缩了很多的特征信息，GAP层就可看作分类器，最后的结果就是分类器结合不同权重的线性叠加，所以CAM就是最后一个卷积层的输出和权重的线性叠加，转变成热力图，放大到和输入图片相同的尺寸，并和原始图片叠加在一起。下面给出本文中CAM的结果。


```python
# 绘制CAM的函数
def show_am(fig_data, change_out, weights, pre_result, label, fig_sign, cl=class_list):
    # 首先获得卷积核的个数
    count = len(change_out)
    # 需要绘制的图片的个数
    count += 3
        
    index = 0
    # 判断概率最大值得索引
    max_index = tf.argmax(pre_result[0])
    # 获得对应的权重
    weight = weights[0][:, max_index]
    bias = weights[1][max_index]
    height, width, c = fig_data.shape
    
    plt.figure(figsize=(18, 9))
    
    # 原始图
    plt.subplot(1, count, 1)
    figdata = np.uint8(255 * fig_data)
    plt.imshow(figdata)
    plt.axis('off')
    plt.title('原始图片类别\n{}'.format(cl[max_index]))
    
    # 绘制卷积核的图
    for out in change_out:
        # 获取对应的权重
        plt.subplot(1, count, 2+index)
        plt.title('第%s个卷积核\n $%.5f \\times $' % (index+1, weight[index]))
        # 绘制热力图
        kk = out[:,:,None] 
        # 归一化【0，1】
        kk = tf.nn.relu(kk)
        kk /= np.max(kk)
        image = tf.image.resize(kk, [height, width])[:,:,0]
        image = np.uint8(255 * image)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        if index != 0:
            plt.ylabel('+',fontsize=19)
        index += 1
    
    # 绘制最终的热力图:现行叠加的
    plt.subplot(1, count, 2+index)
    last_data = tf.add(tf.math.reduce_sum(tf.multiply(change_out,weight[:,None,None]), axis=0), bias)
    last_data = last_data[:, :, None]
    last_data = tf.nn.relu(last_data)
    last_data /= np.max(last_data)
    image = tf.image.resize(last_data, [height, width])[:,:,0]
    image = np.uint8(255 * image)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('=',fontsize=19, verticalalignment='center',horizontalalignment='center',rotation='horizontal')
    plt.title('线性叠加的热力图\n偏置{:.5f}'.format(bias))
    
    # 绘制热力图和原始图片的叠加
    plt.subplot(1, count, 3+index)
    # 伪彩色
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    super_img = heatmap * 0.3 + figdata
    cv2.imwrite('cam.jpg', super_img)
    data = cv2.imread('cam.jpg')
    b,g,r=cv2.split(data)
    image=cv2.merge([r,g,b])
    plt.imshow(image)
    plt.axis('off')
    plt.title('CAM 预测类别\n%s' % (cl[max_index]))
    plt.savefig('cam_%s.png'% fig_sign, dpi=180, bbox_inches='tight')
    plt.show()
```


```python
# 在测试数据集中随机选择n个图片，对GAP层前面的卷积层输出CAM
def get_am(model, data=TEST_DATA, conv_name='CONV_7', gap_name='GAP_1', out_name='OUTPUT', n=10):
    fig_sign = 0
    # 模型
    conv_layer = model.get_layer(conv_name) 
    gap_layer = model.get_layer(gap_name) 
    cam_model =models.Model([model.inputs], [conv_layer.output, gap_layer.output, model.output])
    # 遍历图片
    for d in data.take(n):
        # 数据，标签
        x, label = d
        # 图片数据
        fig_data = x.numpy()
        # 添加维度
        add_fig_data = np.expand_dims(fig_data, axis=0)
        # 卷积层输出、预测结果
        conv_out, gap_out, pre_result = cam_model(add_fig_data)
        # 变换维度
        change_out = tf.transpose(conv_out[0], perm=[2, 0, 1])
        # 获取最后全连接层的参数
        weights = model.get_layer(out_name).weights
        
        # 显示CAM:图片数据、最后卷积层的输出、模型预测结果、图片标签、保存图片的名称
        show_am(fig_data, change_out, weights, pre_result, label, fig_sign)
        
        fig_sign += 1

get_am(model_best, n=5)    
```


![png](output_42_0.png)



![png](output_42_1.png)



![png](output_42_2.png)



![png](output_42_3.png)



![png](output_42_4.png)



  + **(2)加权梯度类激活图（Grad-CAM）**
    

如果训练好的模型的结构不符合要求，则可以利用Grad-CAM的方法去实现，避免了修改模型结构，再次进行训练的麻烦。 Grad-CAM的原理就是给出一张图片，得到模型针对该图片的预测概率的最大值，并计算该值针对卷积层输出的梯度，并计算梯度的均值，将梯度均值与对应的每个卷积核的输出相乘，求和然后计算均值，得到类激活图。

随机在测试数据集中选取用于展示的n张图片，分别针对每个图片输出卷积层的类激活图。


```python
# 绘制grad_cam和原始图片数据的函数
def show_grad_cam(figdata, cam_data, conv_name, figname, savefigname, class_sign, cl=class_list):
    # 首先获取原始数据的维度
    height, width, c = figdata.shape
    # 将热力图数据也变为同样的维度：先保存为图片，在更改维度
    kk = cam_data[0][:,:,None] 
    image = tf.image.resize(kk, [height, width])[:,:,0]

    # 绘制图像：原始图片
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 3, 1)
    figdata = np.uint8(255 * figdata)
    plt.imshow(figdata)
    plt.axis('off')
    plt.title('原始图片类别\n{}'.format(cl[figname]))
    
    # 热力图
    plt.subplot(1, 3, 2)
    image = np.uint8(255 * image)
    plt.imshow(image)
    plt.axis('off')
    plt.title('层{}热力图'.format(conv_name))

    # 叠加
    plt.subplot(1, 3, 3)
    # 伪彩色
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.6 + figdata
    cv2.imwrite('elephant_cam.jpg', superimposed_img)
    data = cv2.imread('elephant_cam.jpg')
    b,g,r=cv2.split(data)
    image=cv2.merge([r,g,b])
    plt.imshow(image)
    plt.axis('off')
    plt.title('GRAD_CAM 预测类别\n%s' % (cl[class_sign]))
      
    plt.savefig('grad_cam_%s.png'% savefigname, dpi=100, bbox_inches='tight')
    plt.show() 
    
```


```python
# 测试数据集中随机选择n张
def get_grad_cam(model, data=TEST_DATA, conv_name=None, n=5):
    """
    model:训练好的模型
    data:测试数据集
    conv_name:默认为最后一个卷积层。注意卷积层名字要正确
    n:随机选择的图片个数
    """
    # 获取所有的卷积层
    conv_layer = get_conv(model)
    # 默认为最后一个卷积层
    if not conv_name:
        conv_name = model.get_layer(sorted(conv_layer.items(), key= lambda s:s[0])[-1][1][0])
    else:
        conv_name = model.get_layer(conv_name)
        
    # 定义计算热力图的模型
    heatmap_model =models.Model([model.inputs], [conv_name.output, model.output])
    # 随机选择一张图片
    fig_sign = 0
    for d in data.take(n):
        # 数据，标签
        x, label = d
        # 图片数据
        fig_data = x.numpy()
        # 添加维度
        add_fig_data = np.expand_dims(fig_data, axis=0)
        
        # 计算梯度
        with tf.GradientTape() as gtape:
            # 卷积层的输出，以及模型针对该图片的输出
            conv_output, pre_result = heatmap_model(add_fig_data)
            # 选择模型的预测值，也就是概率较大的
            class_sign = tf.math.argmax(pre_result[0])
            prob = pre_result[:, class_sign]  
            # 计算概率值对于卷积层输出的梯度
            grads = gtape.gradient(prob, conv_output) 
            # 计算梯度的均值
            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  
        # 输出乘以该卷积核的平均梯度 加在一起计算均值    
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        # 选择值不小于0的
        heatmap = tf.nn.relu(heatmap)
        # 归一化【0，1】
        heatmap /= np.max(heatmap)
        # 图像的展示：图片数据、热力图、卷积层名称、图片标签、保存图片名称、概率结果中最大值索引
        show_grad_cam(fig_data, heatmap, conv_name.name, label, fig_sign, class_sign)
        fig_sign += 1
    return print('Grad-CAM运行结束')

get_grad_cam(model_best, conv_name='CONV_7')
```


![png](output_46_0.png)



![png](output_46_1.png)



![png](output_46_2.png)



![png](output_46_3.png)



![png](output_46_4.png)


    Grad-CAM运行结束
    


```python

```


```python

```


```python

```


```python

```
