import torch
import random
import numpy as np

# for recheck ability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# import data
import torchvision.datasets

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train = X_train.float()
X_test = X_test.float()

import matplotlib.pyplot as plt

"""
plt.imshow(X_train[0, :, :])
plt.show()
print(y_train[0])
"""
X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])

"""
class FCLayer():
    def __init__(self, enters, exits):
        self.weight = torch.rand([exits, enters], requires_grad=True)
        self.bias = torch.rand([exits], requires_grad=True)

    def __call__(self, vector_x):
        return torch.mm(vector_x, self.weight.t()) + self.bias
"""

class Net(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(Net, self).__init__()
        #self.fc1 = FCLayer(28*28, hidden_neurons)
        self.fc1 = torch.nn.Linear(28 * 28, hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        #self.fc2 = FCLayer(hidden_neurons, hidden_neurons)
        self.fc2 = torch.nn.Linear(hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


mnist_net = Net(100)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_net = mnist_net.to(device)

learn_rate = 3.0e-4
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=learn_rate)
batch_size = 150

print(X_train.shape)

test_accuracy_history = []
test_loss_history = []

X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(500):
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds = mnist_net.forward(X_batch)  # calc value of last neurons exit

        loss_value = loss(preds, y_batch)  # calc loss function value
        loss_value.backward()  # calc derivative of loss function

        optimizer.step()  # make a step of gradient descent

    test_preds = mnist_net.forward(X_test)
    test_loss_history.append(loss(test_preds, y_test))

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    test_accuracy_history.append(accuracy)
    #print(accuracy)

"""
weight1 = mnist_net.fc1.weight.data.numpy()
weight2 = mnist_net.fc2.weight.data.numpy()
bias1 = mnist_net.fc1.bias.data.numpy()
bias2 = mnist_net.fc2.bias.data.numpy()
np.savetxt("1.txt", weight1)
np.savetxt("2.txt", weight2)
np.savetxt("3.txt", bias1)
np.savetxt("4.txt", bias2)
"""
#show graphs
plt.plot([loss.detach().numpy() for loss in test_loss_history])
plt.show()
plt.plot([loss.detach().numpy() for loss in test_accuracy_history])
plt.show()
