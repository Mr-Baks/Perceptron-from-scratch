import neural_network
import numpy as np

model = neural_network.NN(784, 64, 256, 10)

# data = np.genfromtxt('emnist-digits-train.csv', delimiter=',')
data = np.genfromtxt('mnist_train.csv', delimiter=',')
np.random.shuffle(data)
X_train = data[:, 1:] / 255
y_train = data[:, :1].flatten().astype(int)
y_train = np.eye(data.shape[0], 10)[y_train]
# print(data)

# data_test = np.genfromtxt('emnist-digits-test.csv', delimiter=',')
data_test = np.genfromtxt('mnist_test.csv', delimiter=',')
X_test = data_test[:, 1:]
y_test = data_test[:, :1]
y_test = np.eye(data_test.shape[0], 10)[data_test[:, :1].flatten().astype(int)]

counter = 0
for i, j in zip(X_test, y_test):
    counter += np.argmax(model.softmax(model.forward(i))) == np.argmax(j)
print('acc', counter)

model.load()
model.train(X_train, y_train, X_test, y_test, epochs=80, learning_rate=0.01, batch_size=256) # <===========================

print(model.hist)

print(model.get_acc(X_test, y_test))

model.save()