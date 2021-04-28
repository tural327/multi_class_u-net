# multi_class_u-net
- [x] First step lib was loading 
- [x] I build a dataset for network 
- [x] Network was built

```jupyter-notebook
inputs = tf.keras.layers.Input((img_h,img_w,img_c))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#network left side of U -net
c1 = tf.keras.layers.Conv2D(16,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)


c2 = tf.keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)


c4 = tf.keras.layers.Conv2D(128,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c5)
p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)
# bottom side of network u net 
c6 = tf.keras.layers.Conv2D(512,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(p5)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(512,(4,4),kernel_initializer='he_normal',padding="same",activation="relu")(c6)

# right side of U net
u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7,c5])
c7 = tf.keras.layers.Conv2D(256,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c7)


u8 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8,c4])
c8 = tf.keras.layers.Conv2D(128,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(128,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c8)

u9 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9,c3])
c9 = tf.keras.layers.Conv2D(64,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(64,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c9)

u10 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding="same")(c9)
u10 = tf.keras.layers.concatenate([u10,c2])
c10 = tf.keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(u10)
c10 = tf.keras.layers.Dropout(0.2)(c10)
c10 = tf.keras.layers.Conv2D(32,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c10)

u11 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding="same")(c10)
u11 = tf.keras.layers.concatenate([u11,c1],axis=3)
c11 = tf.keras.layers.Conv2D(16,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(u11)
c11 = tf.keras.layers.Dropout(0.1)(c11)
c11 = tf.keras.layers.Conv2D(16,(3,3),kernel_initializer='he_normal',padding="same",activation="relu")(c11)

outputs = tf.keras.layers.Conv2D(n_class,(1,1),activation="sigmoid")(c11)
model = tf.keras.Model(inputs = [inputs],outputs=[outputs])
```

- [x] After training I ploted loss and acc score:
- [Acc] ![](https://github.com/tural327/multi_class_u-net/blob/master/acc.png)
- [Loss] ![](https://github.com/tural327/multi_class_u-net/blob/master/loss.png)

# End of testing I made testing while using other testing folder dataset
![](https://github.com/tural327/multi_class_u-net/blob/master/result.png)

