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

# show successful predictions only
win_data = []
for i, item in enumerate(X):
  if predictions[i] == 1 and Y[i] == 1:
    #print('%s => %d (expected %d)' % (item.tolist(), predictions[i], Y[i]))
    win_data.append(list(item))

# sort by price
from operator import itemgetter
win_data = sorted(win_data, key=itemgetter(5),reverse=True)
print "User ID, Age, Sex, Num Titles Owned, Product Id, Product Price"
for i in win_data:
	print(i)
