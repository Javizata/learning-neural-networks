import time

import numpy as np

#We use numpy because its up to 50 times faster than standard lists for numerical computations.
#Data is stored in one continuous block of memory
#We can run math operations directly on numpy arrays without loops

def sigmoid(x):
    """
    Returns a value between 0 and 1 using the sigmoid activation function.
    """
    """
    For many years this was the most commonly used activation function in neural networks. It is particularly useful for models where the output needs to be interpreted as a probability.
    The problem was that it could cause vanishing gradients, making it difficult for the network to learn in deep architectures.
    """
    return 1 / (1 + np.exp(-x))

def reLU(x):
    """
    Returns a value using the ReLU activation function, which outputs 0 for negative inputs and the input itself for non-negative inputs.
    """
    """
    ReLU is widely used in deep learning because it helps mitigate the vanishing gradient problem and allows for faster training.
    """
    return np.maximum(0, x)

def nn_2x(X, W , expected):
    """
    Computes the output of a simple NN given input  X, weight  W.
    
    """
    
    aux =  X * W 
    while abs(expected - aux) > 0.0001:
        previous_aux = aux
        error = (expected - aux)**2
        print(f"Error for output: {error}")
        der = -2 * (expected - aux)
        der = der * X
        print(f"Derivative for output: {der}")
        print()
        new_w = W - 0.01 * der  # Assuming a learning rate of 0.01
        print(f"Updated weights for output: {new_w}")
        #time.sleep(.5)
        W = new_w
        aux = X * W
        print(f"Previous output before weight update: {previous_aux}")
        print(f"New output after weight update: {aux}")
    return W

def nn_batch_gradient_descent(X,W, B, expected):
    """
    Computes the output of a simple NN given input vector X and weight matrix W.
    
    """
    aux =  X * W + B
    print(f"Initial output: {aux}")
    for epoch in range(3000):
        previous_aux = aux
        loss = np.mean((expected - aux)**2)
        print(f"Error for output: {loss}")
        grad_w = np.mean(-2 * (expected - aux) * X)
        grad_b = np.mean(-2 * (expected - aux))
        print(f"Derivative for output: {grad_w}")
        print()
        new_w = W - 0.01 * grad_w  # Assuming a learning rate of 0.01
        new_bias = B - 0.01 * grad_b  # Assuming a learning rate of 0.01
        print(f"Updated weights for output: {new_w}")
        #time.sleep(.5)
        W = new_w
        B = new_bias
        aux = X * W + B
        print(f"Previous output before weight update: {previous_aux}")
        print(f"New output after weight update: {aux}")
        
    return W, B

def nn_xor(X,W_H,W_O,B_H,B_O,expected):
    sample_count = X.shape[0]
    learning_rate = 0.01

    for epoch in range(3000):
        zh  = X @ W_H + B_H
        print(f"Hidden layer linear combination (zh): {zh}")
        print(f"Hidden layer bias (B_H): {B_H}")    
        ah = sigmoid(zh)
        print(f"Hidden layer activation: {ah}")
        zo = ah @ W_O + B_O
        print(f"Output layer linear combination (zo): {zo}")
        print(f"Output layer bias (B_O): {B_O}")
        ao = sigmoid(zo)
        print(f"Output layer activation: {ao}")
        loss = np.mean((expected - ao)**2)
        delta_o = 2 * (ao - expected) * ao * (1 - ao)
        grad = ah.T @ delta_o / sample_count
        grad_b_o = np.mean(delta_o, axis=0, keepdims=True)
        dah = delta_o @ W_O.T
        W_O = W_O - learning_rate * grad
        B_O = B_O - learning_rate * grad_b_o
        print(f"Updated output layer weights: {W_O}")
        print(f"Updated output layer bias: {B_O}")
        print(f"Gradient for output: {grad}")
        print(f"Loss for output: {loss}")
        print(delta_o.shape)
        delta_h = dah * ah * (1 - ah)
        grad_h = X.T @ delta_h / sample_count
        W_H = W_H - learning_rate * grad_h
        B_H = B_H - learning_rate * np.mean(delta_h, axis=0, keepdims=True)
        print(f"Updated hidden layer weights: {W_H}")
        print(f"Updated hidden layer bias: {B_H}")
        print(f"Gradient for hidden layer: {grad_h}")
        print(f"Loss for hidden layer: {np.mean(delta_h**2)}")
        print(f"Epoch: {epoch}")

    ah = sigmoid(X @ W_H + B_H)
    return sigmoid(ah @ W_O + B_O)

if __name__ == "__main__":
    # The operation is matrix multiplication Z = X @ W

    # Input vector X and weight matrix W
    X = np.array([1,2,3,4])
    # Weight matrix W
    W = np.array([[0.2]])
    O = np.array([[0.2,0.5],
                  [0.3,0.8]])
    # Bias vector B
    B = np.array([0.4,0.6])

    # Output vector Z after applying the linear transformation and adding the bias
    #Z = X @ W + B
    #print("Z:", Z)
    #print("sigmoid(Z):", sigmoid(Z))
    #print("reLU(Z):", reLU(Z))
    
    """
    The gradient is a value that indicates how much the output of a function changes with respect to its input. 
    It implies the direction and rate (intensity) of change of the function.
    """
    
    """
    The gradient of the activation functions can be computed for backpropagation purposes.
    For the sigmoid function, the gradient is sigmoid(x) * (1 - sigmoid(x)).
    For the ReLU function, the gradient is 1 for positive inputs and 0 for negative inputs.
    """
    
    #The most important Deep Learning formula is: new_weight = old_weight - learning_rate * gradient
    #Where:
    # - new_weight is the updated weight after applying the gradient descent step.
    # - old_weight is the current weight before the update.
    # - learning_rate is a hyperparameter that controls the step size of the update.
    # - gradient is the derivative of the loss function with respect to the weight.
    
    """
    The derivative is used to calculate the answer to the question If i alter a little this weight the error increases or decreases?.
    The gradient formula is dw/dL​ = ∂L/∂w
    - If the gradient is negative it indicates that increasing the weight will decrease the loss.
    - If the value is big it indicates that a small change in the weight will result in a large change in the loss.
    
    The error function measures how far the network's predictions are from the actual target values.
    The formula for the error function depends on the specific task and loss function used, such as mean squared error for regression or cross-entropy loss for classification.
    In this first case we will use L = (target - prediction)^2
     
    The learning rate is a hyperparameter that controls how much the weights are adjusted during each step of the gradient descent optimization process.
    If the learning rate is too small the training process will be very slow.
    If the learning rate is too large the training process may become unstable and the loss may not converge.
    The most commonly used learning rates are:
        - 0.1
        - 0.01
        - 0.001
     
    """
    
    weight = nn_2x(1, 0.3, 2)
    
    W, B = nn_batch_gradient_descent(X, 0.3, 0.5, np.array([2,4,6,8]))
    print("Updated W:", W)
    print("Updated B:", B)
    print("Updated output Z after training:", X * W + B)

    """
    The gradients are used to update the weights and biases during the training process.
    When we have more than one neuron in a layer, the gradients for each neuron are computed separately, and the weights and biases are updated accordingly.
    
    The gradients are also stored on matrices corresponding to the weights and biases, allowing for efficient updates during the training process.
    Numpy Pytorch and other deep learning frameworks provide efficient implementations for storing and updating gradients during the training process.
    The GPU calculates the gradients and performs the weight updates in parallel, significantly speeding up the training process for large neural networks.
    """
    print(nn_xor(X = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]]),
                W_H = np.array([[0.3, 0.4],[0.2,0.1]]),
        W_O = np.array([[0.3],[0.2]]),
        B_H = np.array([[0.1,0.2]]),
        B_O = np.array([[0.1]]),
        expected=np.array([[0.0],[1.0],[1.0],[0.0]])))
    
    """
    The network may converge to predicting ~0.5 for every sample.
    XOR contains 2 zeros and 2 ones, so the average target value is 0.5.
    Predicting 0.5 everywhere is a "lazy" local solution that gives a stable loss
    without actually learning the XOR decision boundary.
    """
    """
    The hidden layer neurons start with very similar weights.
    When neurons learn nearly identical features, they become redundant and the
    network loses representational power. Initializing weights randomly helps
    break this symmetry and encourages different neurons to learn different patterns.
    """
    
    