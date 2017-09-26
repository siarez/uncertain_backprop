from sklearn.datasets import fetch_mldata
import torch
import numpy as np
from matplotlib import pyplot as plt


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
training_set_size, D_in, H, D_out = 4000, 28 * 28, 20, 10
batch_size = 200
testset_size = 2000

def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))

def dSigmoid(h):
    return np.multiply(h, 1 - h)

# preparing data
mnist = fetch_mldata('MNIST original', data_home='./')


learning_rate = 0.007
epochs = 10
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
training_acc = []

for run in range(30):  # running the experiment multiple time to make sure changes are statistically significant

    # create a list of random indices for training set
    train_idx = np.random.choice(len(mnist.data), training_set_size, replace=False)
    # create x and y by picking samples from the random indices
    mnist_x = np.array([mnist.data[i] for i in train_idx])
    mnist_x = torch.ByteTensor(mnist_x).type(dtype)
    mnist_y = np.array([[mnist.target[i] for i in train_idx]]).transpose()
    mnist_y = torch.DoubleTensor(mnist_y).type(torch.LongTensor)

    # One hot encoding
    y_onehot = torch.zeros([training_set_size, D_out]).type(dtype)
    y_onehot.scatter_(1, mnist_y, 1.0)
    mnist_x /= 255  # scaling down x to fall between 0 and 1
    x = torch.cat((mnist_x, torch.ones([training_set_size, 1])), 1)  # adding biases

    x_batches = torch.split(x, batch_size)
    y_batches = torch.split(mnist_y, batch_size)
    y_onehot_batches = torch.split(y_onehot, batch_size)

    # Randomly initialize weights
    w1 = torch.randn(D_in + 1, H).type(dtype) / np.sqrt(D_in + 1)
    w2 = torch.randn(H + 1, D_out).type(dtype) / np.sqrt(H + 1)

    for t in range(epochs):
        num_of_batches = int(training_set_size / batch_size)
        for b in range(num_of_batches):
            # Forward pass: compute predicted y
            h_logits = x_batches[b].mm(w1)
            h = sigmoid(h_logits)
            h_biased = torch.cat((h, torch.ones([batch_size, 1])), 1)  # adding biases
            y_logits = h_biased.mm(w2)
            y_pred = sigmoid(y_logits)
            if b == num_of_batches - 1:
                # Compute and print loss
                loss = (y_pred - y_onehot_batches[b]).pow(2).sum()
                #print(t, loss)
                _, predicted_classes = torch.max(y_pred, dim=1)
                accuracy = torch.sum(torch.eq(predicted_classes, y_batches[b][:, 0])) / batch_size
                training_acc.append(accuracy)
                #print('batch accuracy: ', accuracy)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            # calculate dLoss/dW2 = h*dh/dh_logits
            delta_y = torch.mul((y_pred - y_onehot_batches[b]), dSigmoid(y_pred))
            grad_w2 = h_biased.t().mm(delta_y)
            # calculating expanded dEdh
            # expanded_dEdh allows us to look at various kinds of distributions which get “summed-away” in backprop.
            delta_y_broadcastable = delta_y.unsqueeze(1)
            expanded_dEdh = torch.mul(delta_y_broadcastable, w2[:-1, :])
            dSig_h_broadcastable = dSigmoid(h).unsqueeze(-1)
            delta_h_expanded = torch.mul(expanded_dEdh, dSig_h_broadcastable)

            if False:
                # This block of code is for tweaking backprop. with information derived from distributions.
                delta_h_sum = torch.sum(delta_h_expanded)
                delta_h_var = torch.std(delta_h_expanded, dim=0)
                adjustment_coef = torch.exp(delta_h_var)
                delta_h_expanded_adjusted = torch.div(delta_h_expanded, adjustment_coef)
                delta_h_expanded_adjusted_sum = torch.sum(delta_h_expanded_adjusted)
                weight_scaling = delta_h_expanded_adjusted_sum / delta_h_sum
                delta_h = torch.sum(delta_h_expanded, dim=2).mul(weight_scaling)
                delta_h = torch.sum(delta_h_expanded_adjusted, dim=2).mul(weight_scaling)
            else:
                # calculates delta like normal backprop
                delta_h = torch.sum(delta_h_expanded, dim=2)  # calculates delta for each heuron in the hidden layer for each training sample

            grad_w1 = x_batches[b].t().mm(delta_h)
            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

            hist_min = -0.025  # torch.min(expanded_dEdh[:, 1, 1])*1.5
            hist_max = 0.025  # torch.max(expanded_dEdh[:, 1, 1])*1.5
            neuron_to_plot = 0  # index of the neuron we want to plot deltas for
            if t == 0 and b == 0 and False:
                delta_h_means = torch.mean(delta_h_expanded, dim=0)  # calculates the batch's average delta for each weight from hidden to output layer
                _, max_index = torch.max(delta_h_means[neuron_to_plot], dim=0)  #for neuron_to_plot hidden layer, it find the index of the weight with max delta
                _, min_index = torch.min(delta_h_means[neuron_to_plot], dim=0)  #for neuron_to_plot in hidden layer, it find the index of the weight with min delta
                ax1.hist(delta_h_expanded[:, neuron_to_plot, max_index[0]].numpy(), bins=101, range=(hist_min, hist_max), histtype='step')
                ax2.hist(delta_h_expanded[:, neuron_to_plot, min_index[0]].numpy(), bins=101, range=(hist_min, hist_max), histtype='step')

    # Starting test
    # picking test data
    # removing samples that are present in training set
    data_training_removed = np.delete(mnist.data, train_idx, 0)
    target_training_removed = np.delete(mnist.target, train_idx, 0)

    test_idx = np.random.choice(len(data_training_removed), testset_size)
    test_x = np.array([data_training_removed[i] for i in test_idx])
    test_x = torch.ByteTensor(test_x).type(dtype)
    test_y = np.array([[target_training_removed[i] for i in test_idx]]).transpose()
    test_y = torch.DoubleTensor(test_y).type(torch.LongTensor)
    test_x = torch.cat((test_x, torch.ones([testset_size, 1])), 1)  # adding biases
    # One hot encoding
    test_y_onehot = torch.zeros([testset_size, D_out]).type(dtype)
    test_y_onehot.scatter_(1, test_y, 1.0)

    # Forward pass: compute predicted y
    h_logits = test_x.mm(w1)
    h = sigmoid(h_logits)
    h_biased = torch.cat((h, torch.ones([testset_size, 1])), 1)  # adding biases
    y_logits = h_biased.mm(w2)
    y_pred = sigmoid(y_logits)

    # Compute and print loss for testset
    _, predicted_classes = torch.max(y_pred, dim=1)
    accuracy = torch.sum(torch.eq(predicted_classes, test_y[:, 0])) / testset_size
    #print('test accuracy: ', accuracy)

training_acc = np.array(training_acc)
print(np.mean(training_acc), np.var(training_acc))

#plt.show()
