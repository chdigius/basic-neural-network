from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#load data
dataset = loadtxt('customer_data_2.csv', delimiter=',')
#split into inpit (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

#define the keras model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit/train the keras model on the dataset
model.fit(X, Y, epochs=25, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(X)
# round predictions 
rounded = [round(x[0]) for x in predictions]
# make class predictions with the model
predictions = model.predict_classes(X)

# summarize the first 5 cases
for i in range(10):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))