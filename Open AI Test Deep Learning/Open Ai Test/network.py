from fc_net import FullyConnectedNet
import matplotlib.pyplot as plt
import numpy as np
from solver import Solver
from sklearn import datasets


def plot_decision_boundary(model, X, y):
    Z = model.loss(X)
    plt.plot(X, Z)
    plt.show()


# Generate a dataset and plot it
X = np.linspace(0, 6.28, 2100)
y = np.sin(X)


data = {}
data.update({'X_train': X[0:2000].reshape(2000, 1)})
data.update({'X_val': X[1500:2000].reshape(500, 1)})
data.update({'y_train': y[0:2000].reshape(2000, 1)})
data.update({'y_val': y[1500:2000].reshape(500, 1)})

num_train = 1900
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

weight_scale = 0.1
learning_rate = 0.01

model = FullyConnectedNet([100, 100],
                          input_dim=1,
                          num_classes=1,
                          weight_scale=weight_scale,
                          dtype=np.float64)

solver = Solver(model,
                small_data,
                print_every=1,
                num_epochs=10000,
                batch_size=1000,
                verbose=True,
                update_rule='rmsprop',
                optim_config={
                  'learning_rate': learning_rate
                })


solver.train()

plt.figure(1)
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.figure(2)
plot_decision_boundary(model, np.linspace(0, 6.28, 2100), y)
plt.show()
