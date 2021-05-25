import numpy as np
from keras.datasets import fashion_mnist
from sklearn.utils import shuffle
import math


class FeedForwardNN:
    def __init__(self, hidden_layer_size, output_layer_size, input_layer_size, learning_rate, num_epochs, optimisation,
                 batch_size, activation_function, min_learning_rate=5e-07, min_delta=0.0005,
                 factor=0.5, patience=3, drop_probability=0):
        np.random.seed(0)
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.input_layer_size = input_layer_size
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        if self.activation_function == "selu":
            weight_initialisation = "lecun"
        else:
            weight_initialisation = "xavier"
        self.initialisation(weight_initialisation)
        self.optimisation = optimisation
        self.batch_size = batch_size
        self.min_learning_rate = min_learning_rate
        self.min_delta = min_delta
        self.factor = factor
        self.patience = patience
        self.drop_probability=drop_probability

    def initialisation(self, weight_initialisation="xavier"):
        n1 = self.input_layer_size
        n2 = self.hidden_layer_size
        n3 = self.output_layer_size
        if weight_initialisation == "xavier":

            self.W1 = np.random.randn(n2, n1) * np.sqrt(2. / (n2 + n1))
            self.b1 = np.zeros((n2, 1))
            self.W2 = np.random.randn(n3, n2) * np.sqrt(2. / (n3 + n2))
            self.b2 = np.zeros((n3, 1))

        elif weight_initialisation == "lecun":

            self.W1 = np.random.randn(n2, n1) * np.sqrt(1. / n1)
            self.b1 = np.zeros((n2, 1))
            self.W2 = np.random.randn(n3, n2) * np.sqrt(1. / n2)
            self.b2 = np.zeros((n3, 1))

        self.a1 = np.zeros((n2, 1))
        self.h1 = np.zeros((n2, 1))
        self.a2 = np.zeros((n3, 1))
        self.h2 = np.zeros((n3, 1))

    def activation(self, X, derivative=False):
        s = self.activation_function

        if s == "relu":
            if derivative == False:
                return np.where(X > 0, X, 0)
            return np.where(X > 0, 1, 0)

        elif s == "lrelu":
            if derivative == False:
                return np.where(X > 0, X, 0.2 * X)
            return np.where(X > 0, 1, 0.2)

        elif s == "selu":
            alpha = 1.67326
            lamd = 1.05070
            if derivative == False:
                return np.where(X > 0, lamd * X, lamd * alpha * (np.exp(X) - 1))
            return np.where(X > 0, lamd, lamd * alpha * np.exp(X))

    def output(self, X):
        exp = np.exp(X - X.max())
        return (exp / np.sum(exp, axis=0))

    def forward_pass(self, X):
        self.a1 = self.b1 + (self.W1 @ X)
        self.h1 = self.activation(self.a1)
        self.h1= self.dropout(self.h1)
        self.a2 = self.b2 + (self.W2 @ self.h1)
        self.h2 = self.output(self.a2)

    def backward_pass(self, X, y):
        e = np.zeros((10, 1))
        e[y] = 1
        temp = -(e - self.h2)
        grad_w2 = temp @ (self.h1.T)
        grad_b2 = temp
        temp = (self.W2).T @ temp
        temp = np.multiply(temp, self.activation(self.a1, derivative=True))
        grad_w1 = temp @ X.T
        grad_b1 = temp

        return (grad_w1, grad_b1, grad_w2, grad_b2)

    def validate(self, X_test, y_test, alpha):
        count = 0
        loss = 0
        for x, y in zip(X_test, y_test):
            self.forward_pass(x.reshape((x.shape[0], 1)))
            output = self.h2
            e = np.zeros((10, 1))
            e[y] = 1
            loss = loss + np.sum((e - output) ** 2)
            y_predicted = np.argmax(output)
            if y_predicted == y:
                count = count + 1
        loss = loss + (alpha * np.sum(self.W1 ** 2) / 2) + (alpha * np.sum(self.W2 ** 2) / 2)
        return (count / X_test.shape[0], loss / X_test.shape[0])

    def adam(self, X_train, y_train, X_val, y_val, alpha, eps=1e-8, beta1=0.9, beta2=0.999):

        N = X_train.shape[0]
        v_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        v_b1 = np.zeros((self.hidden_layer_size, 1))
        m_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        m_b1 = np.zeros((self.hidden_layer_size, 1))

        v_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        v_b2 = np.zeros((self.output_layer_size, 1))
        m_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        m_b2 = np.zeros((self.output_layer_size, 1))
        best_val_acc, no_inc = 0, 0
        step = 1
        for i in range(self.num_epochs):
            grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
            grad_b1 = np.zeros((self.hidden_layer_size, 1))
            grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
            grad_b2 = np.zeros((self.output_layer_size, 1))
            count = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            for x, y in zip(X_train, y_train):
                x = x.reshape((x.shape[0], 1))
                self.forward_pass(x)
                (temp1, temp2, temp3, temp4) = self.backward_pass(x, y)
                grad_w1 += temp1
                grad_b1 += temp2
                grad_w2 += temp3
                grad_b2 += temp4

                count = count + 1
                if count % self.batch_size == 0 or count == N:
                    v_w1 = beta2 * v_w1 + (1 - beta2) * grad_w1 ** 2
                    v_b1 = beta2 * v_b1 + (1 - beta2) * grad_b1 ** 2
                    v_w2 = beta2 * v_w2 + (1 - beta2) * grad_w2 ** 2
                    v_b2 = beta2 * v_b2 + (1 - beta2) * grad_b2 ** 2

                    m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
                    m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1
                    m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2
                    m_b2 = beta1 * m_b2 + (1 - beta1) * grad_b2

                    m_w1_hat = m_w1 / (1 - math.pow(beta1, step))
                    m_b1_hat = m_b1 / (1 - math.pow(beta1, step))
                    m_w2_hat = m_w2 / (1 - math.pow(beta1, step))
                    m_b2_hat = m_b2 / (1 - math.pow(beta1, step))

                    v_w1_hat = v_w1 / (1 - math.pow(beta2, step))
                    v_b1_hat = v_b1 / (1 - math.pow(beta2, step))
                    v_w2_hat = v_w2 / (1 - math.pow(beta2, step))
                    v_b2_hat = v_b2 / (1 - math.pow(beta2, step))

                    self.W1 = (1 - self.learning_rate * alpha) * self.W1 - (
                            self.learning_rate / (np.sqrt(v_w1_hat) + eps)) * m_w1_hat
                    self.b1 = self.b1 - (self.learning_rate / (np.sqrt(v_b1_hat) + eps)) * m_b1_hat
                    self.W2 = (1 - self.learning_rate * alpha) * self.W2 - (
                            self.learning_rate / (np.sqrt(v_w2_hat) + eps)) * m_w2_hat
                    self.b2 = self.b2 - (self.learning_rate / (np.sqrt(v_b2_hat) + eps)) * m_b2_hat

                    grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
                    grad_b1 = np.zeros((self.hidden_layer_size, 1))
                    grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
                    grad_b2 = np.zeros((self.output_layer_size, 1))
                    step = step + 1
            (train_acc, train_loss) = self.validate(X_train, y_train, alpha)
            (val_acc, val_loss) = self.validate(X_val, y_val, alpha)
            if val_acc - best_val_acc < self.min_delta:
                no_inc += 1
                if no_inc > self.patience:
                    no_inc = 0
                    temp = self.learning_rate * self.factor
                    self.learning_rate = max(temp, self.min_learning_rate)
            else:
                best_val_acc = val_acc

        print("train_acc:", train_acc, " ", "train_loss:", train_loss)
        print("val_acc:", val_acc, " ", "val_loss:", val_loss)
        print("best_val_acc:", best_val_acc)

    def nadam(self, X_train, y_train, X_val, y_val, alpha, eps=1e-7, beta1=0.9, beta2=0.999):

        N = X_train.shape[0]
        v_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        v_b1 = np.zeros((self.hidden_layer_size, 1))
        m_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        m_b1 = np.zeros((self.hidden_layer_size, 1))

        v_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        v_b2 = np.zeros((self.output_layer_size, 1))
        m_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        m_b2 = np.zeros((self.output_layer_size, 1))
        best_val_acc, no_inc = 0, 0
        step = 1
        for i in range(self.num_epochs):
            grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
            grad_b1 = np.zeros((self.hidden_layer_size, 1))
            grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
            grad_b2 = np.zeros((self.output_layer_size, 1))
            count = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            for x, y in zip(X_train, y_train):
                x = x.reshape((x.shape[0], 1))
                self.forward_pass(x)
                (temp1, temp2, temp3, temp4) = self.backward_pass(x, y)
                grad_w1 += temp1
                grad_b1 += temp2
                grad_w2 += temp3
                grad_b2 += temp4
                count = count + 1
                if count % self.batch_size == 0 or count == N:
                    v_w1 = beta2 * v_w1 + (1 - beta2) * grad_w1 ** 2
                    v_b1 = beta2 * v_b1 + (1 - beta2) * grad_b1 ** 2
                    v_w2 = beta2 * v_w2 + (1 - beta2) * grad_w2 ** 2
                    v_b2 = beta2 * v_b2 + (1 - beta2) * grad_b2 ** 2

                    m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
                    m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1
                    m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2
                    m_b2 = beta1 * m_b2 + (1 - beta1) * grad_b2

                    m_w1_hat = m_w1 / (1 - math.pow(beta1, step))
                    m_w1_hat = beta1 * m_w1_hat + ((1 - beta1) * grad_w1) / (1 - math.pow(beta1, step))

                    m_b1_hat = m_b1 / (1 - math.pow(beta1, step))
                    m_b1_hat = beta1 * m_b1_hat + ((1 - beta1) * grad_b1) / (1 - math.pow(beta1, step))

                    m_w2_hat = m_w2 / (1 - math.pow(beta1, step))
                    m_w2_hat = beta1 * m_w2_hat + ((1 - beta1) * grad_w2) / (1 - math.pow(beta1, step))

                    m_b2_hat = m_b2 / (1 - math.pow(beta1, step))
                    m_b2_hat = beta1 * m_b2_hat + ((1 - beta1) * grad_b2) / (1 - math.pow(beta1, step))

                    v_w1_hat = v_w1 / (1 - math.pow(beta2, step))
                    v_b1_hat = v_b1 / (1 - math.pow(beta2, step))
                    v_w2_hat = v_w2 / (1 - math.pow(beta2, step))
                    v_b2_hat = v_b2 / (1 - math.pow(beta2, step))

                    self.W1 = (1 - self.learning_rate * alpha) * self.W1 - (
                            self.learning_rate / (np.sqrt(v_w1_hat) + eps)) * m_w1_hat
                    self.b1 = self.b1 - (self.learning_rate / (np.sqrt(v_b1_hat) + eps)) * m_b1_hat
                    self.W2 = (1 - self.learning_rate * alpha) * self.W2 - (
                            self.learning_rate / (np.sqrt(v_w2_hat) + eps)) * m_w2_hat
                    self.b2 = self.b2 - (self.learning_rate / (np.sqrt(v_b2_hat) + eps)) * m_b2_hat

                    grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
                    grad_b1 = np.zeros((self.hidden_layer_size, 1))
                    grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
                    grad_b2 = np.zeros((self.output_layer_size, 1))
                    step = step + 1
            (train_acc, train_loss) = self.validate(X_train, y_train, alpha)
            (val_acc, val_loss) = self.validate(X_val, y_val, alpha)
            if val_acc - best_val_acc < self.min_delta:
                no_inc += 1
                if no_inc > self.patience:
                    no_inc = 0
                    temp = self.learning_rate * self.factor
                    self.learning_rate = max(temp, self.min_learning_rate)
            else:
                best_val_acc = val_acc

        print("train_acc:", train_acc, " ", "train_loss:", train_loss)
        print("val_acc:", val_acc, " ", "val_loss:", val_loss)
        print("best_val_acc:", best_val_acc)

    def nesterov(self, X_train, y_train, X_val, y_val, alpha, gamma=0.9):

        N = X_train.shape[0]
        prev_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        prev_b1 = np.zeros((self.hidden_layer_size, 1))
        prev_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        prev_b2 = np.zeros((self.output_layer_size, 1))
        best_val_acc, no_inc = 0, 0
        for i in range(self.num_epochs):
            grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
            grad_b1 = np.zeros((self.hidden_layer_size, 1))
            grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
            grad_b2 = np.zeros((self.output_layer_size, 1))
            count = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            for x, y in zip(X_train, y_train):
                x = x.reshape((x.shape[0], 1))
                self.forward_pass(x)
                (temp1, temp2, temp3, temp4) = self.backward_pass(x, y)
                grad_w1 += temp1
                grad_b1 += temp2
                grad_w2 += temp3
                grad_b2 += temp4
                count = count + 1
                if count % self.batch_size == 0 or count == N:
                    prev_w1 = (gamma * prev_w1) + (self.learning_rate * grad_w1)
                    prev_b1 = (gamma * prev_b1) + (self.learning_rate * grad_b1)
                    prev_w2 = (gamma * prev_w2) + (self.learning_rate * grad_w2)
                    prev_b2 = (gamma * prev_b2) + (self.learning_rate * grad_b2)
                    self.W1 = (1 - self.learning_rate * alpha) * self.W1 - (
                            gamma * prev_w1 + self.learning_rate * grad_w1)
                    self.b1 = self.b1 - (gamma * prev_b1 + self.learning_rate * grad_b1)
                    self.W2 = (1 - self.learning_rate * alpha) * self.W2 - (
                            gamma * prev_w2 + self.learning_rate * grad_w2)
                    self.b2 = self.b2 - (gamma * prev_b2 + self.learning_rate * grad_b2)
                    grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
                    grad_b1 = np.zeros((self.hidden_layer_size, 1))
                    grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
                    grad_b2 = np.zeros((self.output_layer_size, 1))

            (train_acc, train_loss) = self.validate(X_train, y_train, alpha)
            (val_acc, val_loss) = self.validate(X_val, y_val, alpha)
            if val_acc - best_val_acc < self.min_delta:
                no_inc += 1
                if no_inc > self.patience:
                    no_inc = 0
                    temp = self.learning_rate / self.factor
                    self.learning_rate = max(temp, self.min_learning_rate)
            else:
                best_val_acc = val_acc
        print("train_acc:", train_acc, " ", "train_loss:", train_loss)
        print("val_acc:", val_acc, " ", "val_loss:", val_loss)
        print("best_val_acc:", best_val_acc)


    def rmsprop(self, X_train, y_train, X_val, y_val, alpha, eps=1e-8, beta=0.9):

        N = X_train.shape[0]
        v_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
        v_b1 = np.zeros((self.hidden_layer_size, 1))

        v_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
        v_b2 = np.zeros((self.output_layer_size, 1))

        best_val_acc, no_inc = 0, 0

        for i in range(self.num_epochs):
            grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
            grad_b1 = np.zeros((self.hidden_layer_size, 1))
            grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
            grad_b2 = np.zeros((self.output_layer_size, 1))
            count = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            for x, y in zip(X_train, y_train):
                x = x.reshape((x.shape[0], 1))
                self.forward_pass(x)
                (temp1, temp2, temp3, temp4) = self.backward_pass(x, y)
                grad_w1 += temp1
                grad_b1 += temp2
                grad_w2 += temp3
                grad_b2 += temp4
                count = count + 1
                if count % self.batch_size == 0 or count == N:
                    v_w1 = beta * v_w1 + (1 - beta) * grad_w1 ** 2
                    v_b1 = beta * v_b1 + (1 - beta) * grad_b1 ** 2
                    v_w2 = beta * v_w2 + (1 - beta) * grad_w2 ** 2
                    v_b2 = beta * v_b2 + (1 - beta) * grad_b2 ** 2

                    self.W1 = (1 - self.learning_rate * alpha) * self.W1 - (
                            self.learning_rate / (np.sqrt(v_w1) + eps)) * grad_w1
                    self.b1 = self.b1 - (self.learning_rate / (np.sqrt(v_b1) + eps)) * grad_b1
                    self.W2 = (1 - self.learning_rate * alpha) * self.W2 - (
                            self.learning_rate / (np.sqrt(v_w2) + eps)) * grad_w2
                    self.b2 = self.b2 - (self.learning_rate / (np.sqrt(v_b2) + eps)) * grad_b2

                    grad_w1 = np.zeros((self.hidden_layer_size, self.input_layer_size))
                    grad_b1 = np.zeros((self.hidden_layer_size, 1))
                    grad_w2 = np.zeros((self.output_layer_size, self.hidden_layer_size))
                    grad_b2 = np.zeros((self.output_layer_size, 1))

            (train_acc, train_loss) = self.validate(X_train, y_train, alpha)
            (val_acc, val_loss) = self.validate(X_val, y_val, alpha)
            if val_acc - best_val_acc < self.min_delta:
                no_inc += 1
                if no_inc > self.patience:
                    no_inc = 0
                    temp = self.learning_rate * self.factor
                    self.learning_rate = max(temp, self.min_learning_rate)
            else:
                best_val_acc = val_acc

        print("train_acc:", train_acc, " ", "train_loss:", train_loss)
        print("val_acc:", val_acc, " ", "val_loss:", val_loss)
        print("best_val_acc:",best_val_acc)

    def fit(self, X_train, y_train, X_val, y_val, alpha):
        s = self.optimisation

        if s == "nesterov":
            self.nesterov(X_train, y_train, X_val, y_val, alpha)
        elif s == "adam":
            self.adam(X_train, y_train, X_val, y_val, alpha)
        elif s == "rmsprop":
            self.rmsprop(X_train, y_train, X_val, y_val, alpha)
        elif s == "nadam":
            self.nadam(X_train, y_train, X_val, y_val, alpha)

    def dropout(self, X):
        drop_probability = self.drop_probability
        keep_probability = 1 - drop_probability
        mask = np.random.uniform(size=X.shape) < keep_probability

        if keep_probability > 0.0:
            scale = (1 / keep_probability)
        else:
            scale = 0.0
        return mask * X * scale

class RBMContrastiveDivergence:
    def __init__(self, num_visible_nodes, num_hidden_nodes, k, learning_rate, num_epochs):
        self.num_visible_nodes = num_visible_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.k = k
        self.W = np.random.randn(num_visible_nodes, num_hidden_nodes) * np.sqrt(
            2. / (num_hidden_nodes + num_visible_nodes))
        self.c = np.zeros((1, num_hidden_nodes))
        self.b = np.zeros((1, num_visible_nodes))
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def activation(self, X):
        temp = np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))
        return temp

    def output(self, X):
        exp = np.exp(X - X.max())
        return (exp / np.sum(exp, axis=0))

    def sample_v(self, h):
        temp = h @ self.W.T
        temp = temp + self.b
        prob_v_given_h = self.activation(temp)
        return (np.random.uniform(size=prob_v_given_h.shape) < prob_v_given_h) * 1

    def sample_h(self, v):
        temp = v @ self.W
        temp = temp + self.c
        prob_h_given_v = self.activation(temp)
        return (np.random.uniform(size=prob_h_given_v.shape) < prob_h_given_v) * 1

    def update_parameters(self, update_W, update_b, update_c):
        self.W += self.learning_rate * update_W
        self.b += self.learning_rate * update_b
        self.c += self.learning_rate * update_c

def load_dataset():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 784))
    X_train = X_train.astype('float64')
    X_train = np.where(X_train >= 127.0, 1, 0)
    X_val, y_val = X_train[48000:], y_train[48000:]
    X_train, y_train = X_train[:48000], y_train[:48000]
    X_test = X_test.reshape((X_test.shape[0], 784))
    X_test = X_test.astype('float64')
    X_test = np.where(X_test >= 127.0, 1, 0)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def train(num_visible_nodes, num_hidden_nodes, k, learning_rate, num_epochs , drop_probability=0):
    X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()
    rbm = RBMContrastiveDivergence(num_visible_nodes=num_visible_nodes, num_hidden_nodes=num_hidden_nodes, k=k,
                                   learning_rate=learning_rate, num_epochs=num_epochs)
    for i in range(rbm.num_epochs):
        (X_train, y_train) = shuffle(X_train, y_train, random_state=0)
        for x in X_train:
            x = x.reshape((1, 784))
            v = x
            h = np.zeros((1, rbm.num_hidden_nodes))
            update_W = np.zeros((rbm.num_visible_nodes, rbm.num_hidden_nodes))
            update_c = np.zeros((1, rbm.num_hidden_nodes))
            update_b = np.zeros((1, rbm.num_visible_nodes))
            for t in range(0, rbm.k):
                h = rbm.sample_h(v)
                v = rbm.sample_v(h)
            act = rbm.activation((v @ rbm.W) + rbm.c)
            temp = rbm.activation((x @ rbm.W) + rbm.c)
            update_W = (x.T @ temp) - (v.T @ act)
            update_b = x - v
            update_c = temp - act
            rbm.update_parameters(update_W, update_b, update_c)
        classifier_input = rbm.sample_h(X_val)
        test_input = rbm.sample_h(X_test)
        classifier = FeedForwardNN(hidden_layer_size=rbm.num_hidden_nodes,output_layer_size=10,
                                   input_layer_size=rbm.num_hidden_nodes, learning_rate=0.001, num_epochs=10,
                                   optimisation="adam", batch_size=64,
                                   activation_function="lrelu",drop_probability=drop_probability)
        classifier.fit(classifier_input, y_val, test_input, y_test, 0.001)

num_visible_nodes = 784
num_hidden_nodes = 256
num_epochs = 1
k = 2
learning_rate = 0.01
train(num_hidden_nodes=num_hidden_nodes,num_visible_nodes=num_visible_nodes,learning_rate=learning_rate,num_epochs=num_epochs,k=k,drop_probability=0)
