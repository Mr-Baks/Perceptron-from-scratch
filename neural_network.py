import numpy as np
import json

class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size): # Инициализация весов и смещений
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2./hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2./hidden_size2)
        self.b3 = np.zeros((1, output_size))

        self.hist = {}
        
        self.dropout_rate = 0.3
    
    # Математические функции
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        if np.sum(np.isnan(x)): print(np.isnan(x)) 
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def dropout(self, x, rate, training): # Выключение нейрона
        if not training:
            return x 
        mask = np.random.binomial(1, 1-rate, size=x.shape) / (1-rate)
        return x * mask
    
    def forward(self, X, training=False): # Прямое распространение, получение предсказания модели
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1 = self.dropout(self.a1, self.dropout_rate, training)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.a2 = self.dropout(self.a2, self.dropout_rate, training)  
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.softmax(self.z3)
    
    def backward(self, X, y, output, learning_rate): # Обратное распространение ошибки: рассче градиентов и обновление весов
        m = X.shape[0]
        
        delta3 = output - y
        dW3 = np.dot(self.a2.T, delta3) / m
        db3 = np.sum(delta3, axis=0, keepdims=True) / m
        
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, X_test, y_test, epochs=100, learning_rate=0.01, batch_size=64): # Тренировка нейросети
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                output = self.forward(X_batch, training=True) 
                self.backward(X_batch, y_batch, output, learning_rate)
            
            output = self.forward(X, training=True) 
            loss = -np.mean(y * np.log(output + 1e-8))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            self.hist[epoch] = (self.get_acc(X_test, y_test))
            print(self.hist[epoch])

    def get_acc(self, X_test, y_test): # Получение количества правильных ответов модели
        counter = 0
        for i, j in zip(X_test, y_test):
          counter += np.argmax(self.softmax(self.forward(i))) == np.argmax(j)
        return int(counter) / X_test.shape[0]

    def save(self): # Сохранение параметров
        with open('params.json', 'w') as p:
            json.dump({
            'W1':self.W1.tolist(), 
            'W2':self.W2.tolist(),
            'W3':self.W3.tolist(),
            'b1':self.b1.tolist(),
            'b2':self.b2.tolist(),
            'b3':self.b3.tolist(),}, p, indent=4)

    def load(self): # Загрузка параметров из params.json
        with open('params.json', 'r') as p:
            params = json.load(p)
            self.W1 = np.array(params['W1'])
            self.b1 = np.array(params['b1'])
            self.W2 = np.array(params['W2'])
            self.b2 = np.array(params['b2'])
            self.W3 = np.array(params['W3'])
            self.b3 = np.array(params['b3'])

