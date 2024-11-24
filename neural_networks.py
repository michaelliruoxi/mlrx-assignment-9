import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Activation function
        self.activation_map = {
            'tanh': (np.tanh, lambda a: 1 - a ** 2),
            'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
            'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda a: a * (1 - a))
        }

        # Weight initialization based on activation function
        if self.activation_fn == 'relu':
            # He initialization for ReLU
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        else:
            # Xavier initialization for tanh and sigmoid
            limit_W1 = np.sqrt(6 / (input_dim + hidden_dim))
            self.W1 = np.random.uniform(-limit_W1, limit_W1, (input_dim, hidden_dim))
            limit_W2 = np.sqrt(6 / (hidden_dim + output_dim))
            self.W2 = np.random.uniform(-limit_W2, limit_W2, (hidden_dim, output_dim))

        # Initialize biases to zeros
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))

        # Placeholders for activations and gradients
        self.activations = None
        self.gradients = None
        
    def forward(self, X):
        # Input to hidden layer
        z1 = np.dot(X, self.W1) + self.b1  # Linear combination
        a1 = self.activation_map[self.activation_fn][0](z1)  # Apply activation function

        # Hidden to output layer
        z2 = np.dot(a1, self.W2) + self.b2  # Linear combination for logits

        # Apply sigmoid activation to output layer
        y_pred = 1 / (1 + np.exp(-z2))
        print(f"Step: Mean activations per neuron: {np.mean(a1, axis=0)}")

        # Store activations for visualization
        self.activations = {
            'X': X,      # Input features
            'z1': z1,    # Linear combination in hidden layer
            'a1': a1,    # Activation in hidden layer
            'z2': z2,    # Logits in output layer
            'y_pred': y_pred  # Activation in output layer
        }

        return y_pred

    def backward(self, X, y):
        # Retrieve stored activations from the forward pass
        z1 = self.activations['z1']
        a1 = self.activations['a1']
        y_pred = self.activations['y_pred']

        # Compute the gradient of the loss w.r.t. the output layer
        m = y.shape[0]  # Number of samples
        dz2 = (y_pred - y)  # Gradient of loss w.r.t. z2
        dW2 = np.dot(a1.T, dz2) / m  # Gradient of weights for hidden to output
        db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradient of biases for output layer

        # Compute the gradient of the loss w.r.t. the hidden layer
        da1 = np.dot(dz2, self.W2.T)  # Backpropagate through weights of output layer

        if self.activation_fn == 'relu':
            dz1 = da1 * self.activation_map[self.activation_fn][1](z1)
        else:
            dz1 = da1 * self.activation_map[self.activation_fn][1](a1)

        dW1 = np.dot(X.T, dz1) / m  # Gradient of weights for input to hidden
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient of biases for hidden layer

        # Update weights and biases using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }



def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y


def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward functions
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Generate a grid in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_grid, yy_grid = np.meshgrid(np.linspace(x_min, x_max, 50),
                                   np.linspace(y_min, y_max, 50))
    grid = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # Compute hidden layer activations for the grid
    z1_grid = np.dot(grid, mlp.W1) + mlp.b1
    a1_grid = mlp.activation_map[mlp.activation_fn][0](z1_grid)

    # Reshape activations to match the grid shape
    if a1_grid.shape[1] >= 3:
        X_hidden_grid = a1_grid[:, 0].reshape(xx_grid.shape)
        Y_hidden_grid = a1_grid[:, 1].reshape(xx_grid.shape)
        Z_hidden_grid = a1_grid[:, 2].reshape(xx_grid.shape)

        # Plot the transformed grid in hidden layer space
        ax_hidden.plot_surface(X_hidden_grid, Y_hidden_grid, Z_hidden_grid,
                               color='lightblue', alpha=0.2, rstride=1, cstride=1,
                               linewidth=0, antialiased=False)

        # Plot hidden features (activations from the hidden layer)
        hidden_features = mlp.activations['a1']  # Hidden layer activations
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                          c=y.ravel(), cmap='bwr', alpha=0.7)

        # Plot the decision plane in the hidden space
        W2 = mlp.W2
        b2 = mlp.b2

        # Check if W2[2, 0] is not close to zero
        if np.abs(W2[2, 0]) > 1e-5:
            # Define a grid for the plane
            x_plane_range = np.linspace(X_hidden_grid.min(), X_hidden_grid.max(), 10)
            y_plane_range = np.linspace(Y_hidden_grid.min(), Y_hidden_grid.max(), 10)
            xx_plane, yy_plane = np.meshgrid(x_plane_range, y_plane_range)

            # Compute z values for the plane
            zz_plane = -(W2[0, 0] * xx_plane + W2[1, 0] * yy_plane + b2[0, 0]) / W2[2, 0]

            # Plot the decision plane
            ax_hidden.plot_surface(xx_plane, yy_plane, zz_plane, alpha=0.3, color='orange')
        else:
            ax_hidden.text(0, 0, 0, "Decision plane undefined (W2[2, 0] ≈ 0)", color='red')

        # Set axis limits to ensure visibility
        ax_hidden.set_xlim(X_hidden_grid.min(), X_hidden_grid.max())
        ax_hidden.set_ylim(Y_hidden_grid.min(), Y_hidden_grid.max())
        ax_hidden.set_zlim(Z_hidden_grid.min(), Z_hidden_grid.max())
        ax_hidden.set_title(f'Hidden Layer Activations at Step {frame * 10}')
        ax_hidden.set_xlabel('Neuron 1')
        ax_hidden.set_ylabel('Neuron 2')
        ax_hidden.set_zlabel('Neuron 3')
    else:
        ax_hidden.text(0.5, 0.5, 0.5,
                       "Not enough dimensions in hidden layer for 3D plot", color='red')
    
    
    # Plot decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_input, yy_input = np.meshgrid(np.linspace(x_min, x_max, 200),
                                     np.linspace(y_min, y_max, 200))
    grid = np.c_[xx_input.ravel(), yy_input.ravel()]
    # Plot decision boundary in input space
    probs = mlp.forward(grid)
    probs = probs.reshape(xx_input.shape)
    ax_input.contourf(xx_input, yy_input, probs, levels=[0, 0.5, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f'Decision Boundary at Step {frame * 10}')

    
    


    # Visualize network gradients as nodes and edges
    # Define positions for nodes
    input_nodes = [(-1, 0.5), (-1, -0.5)]
    hidden_nodes = [(0, 1), (0, 0), (0, -1)]
    output_node = (1, 0)
    # Draw input nodes
    for i, pos in enumerate(input_nodes):
        circle = Circle(pos, 0.1, color='lightblue', ec='k')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0], pos[1], f"X{i+1}", ha='center', va='center')
    # Draw hidden nodes
    for i, pos in enumerate(hidden_nodes):
        circle = Circle(pos, 0.1, color='lightgreen', ec='k')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0], pos[1], f"H{i+1}", ha='center', va='center')
    # Draw output node
    circle = Circle(output_node, 0.1, color='lightcoral', ec='k')
    ax_gradient.add_patch(circle)
    ax_gradient.text(output_node[0], output_node[1], "Y", ha='center', va='center')
    # Draw edges with thickness proportional to gradients
    # Input to hidden weights
    max_grad_W1 = np.max(np.abs(mlp.gradients['dW1']))
    for i, inp_pos in enumerate(input_nodes):
        for j, hid_pos in enumerate(hidden_nodes):
            grad = np.abs(mlp.gradients['dW1'][i, j])
            lw = (grad / max_grad_W1) * 5 if max_grad_W1 != 0 else 1  # Normalize line width
            ax_gradient.plot([inp_pos[0], hid_pos[0]], [inp_pos[1], hid_pos[1]],
                             'k-', lw=lw)
    # Hidden to output weights
    max_grad_W2 = np.max(np.abs(mlp.gradients['dW2']))
    for i, hid_pos in enumerate(hidden_nodes):
        grad = np.abs(mlp.gradients['dW2'][i, 0])
        lw = (grad / max_grad_W2) * 5 if max_grad_W2 != 0 else 1  # Normalize line width
        ax_gradient.plot([hid_pos[0], output_node[0]], [hid_pos[1], output_node[1]],
                         'k-', lw=lw)
    ax_gradient.set_xlim(-1.5, 1.5)
    ax_gradient.set_ylim(-1.5, 1.5)
    ax_gradient.set_title(f'Network Gradients at Step {frame * 10}')
    ax_gradient.axis('off')




def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)