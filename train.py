from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
len(dataset)
train , test = dataset
len(train)
X_train , y_train = train
X_train.shape
X_test , y_test = test
X_test.shape
img1 = X_train[7]
img1.shape
img1_label = y_train[7]
img1_label
img1.shape
import matplotlib.pyplot as plt
img1.shape
img1_1d = img1.reshape(28*28)
img1_1d.shape
X_train.shape
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train_1d.shape
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
X_train.shape
y_train.shape
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_train_cat
y_train_cat[7]
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
h = model.fit(X_train, y_train_cat, epochs=2)






