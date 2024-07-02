#Nв1 = 5 (Определение цены недвижимости, Boston Housing price regression dataset)
#Nв2 = 1 (3 слоя, Dropout, Adam)
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(path="boston_housing.npz", test_split=0.2)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='Adam', loss='mse', metrics=['mse'])
model.summary()

model.fit(x_train, y_train, epochs=10, validation_split=0.1)
n = 0
pred = model.predict(x_test)
for i in range(len(y_test)):
    #print(pred[i], ' ', y_test[i])
    if abs(y_test[i] - pred[i]) < 10:
        n += 1
print(n, ' ', len(y_test))
print(round(n / len(y_test) * 100, 5), '%')

x = numpy.array([[0.02177, 82.5, 2.03, 0, 0.415, 7.61, 15.7, 6.27, 2, 348, 14.7, 395.38, 3.11]])
y = 42.3
pred = model.predict(x)
print(pred, ' ', y)