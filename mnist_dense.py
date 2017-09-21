from sklearn.datasets import fetch_mldata
import torch
import numpy as np
from matplotlib import pyplot as plt


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
training_set_size, D_in, H, D_out = 400, 28 * 28, 20, 10
batch_size = 200
testset_size = 200

def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))

def dSigmoid(h):
    return np.multiply(h, 1 - h)

# preparing data
mnist = fetch_mldata('MNIST original', data_home='./')
# create a list of random indices for traing set
train_idx = np.random.choice(len(mnist.data), training_set_size, replace=False)
# create x and y by picking samples from the random indices
mnist_x = np.array([mnist.data[i] for i in train_idx])
mnist_x = torch.ByteTensor(mnist_x).type(dtype)
mnist_y = np.array([[mnist.target[i] for i in train_idx]]).transpose()
mnist_y = torch.DoubleTensor(mnist_y).type(torch.LongTensor)
print(mnist.data.shape)
print(mnist_x.shape)

# One hot encoding
y_onehot = torch.zeros([training_set_size, D_out]).type(dtype)
y_onehot.scatter_(1, mnist_y, 1.0)

x = torch.cat((mnist_x, torch.ones([training_set_size, 1])), 1)  # adding biases

x_batches = torch.split(x, batch_size)
y_batches = torch.split(mnist_y, batch_size)
y_onehot_batches = torch.split(y_onehot, batch_size)

# Randomly initialize weights
w1 = torch.randn(D_in + 1, H).type(dtype)
w2 = torch.randn(H + 1, D_out).type(dtype)

learning_rate = 0.007
epochs = 10
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

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
            # print('batch #: ', b)
            # Compute and print loss
            loss = (y_pred - y_onehot_batches[b]).pow(2).sum()
            print(t, loss)
            _, predicted_classes = torch.max(y_pred, dim=1)
            accuracy = torch.sum(torch.eq(predicted_classes, y_batches[b][:,0])) / batch_size
            print('batch accuracy: ', accuracy)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        # calculate dLoss/dW2 = h*dh/dh_logits
        delta_y = torch.mul((y_pred - y_onehot_batches[b]), dSigmoid(y_pred))
        grad_w2 = h_biased.t().mm(delta_y)
        # calculating expanded dEdh
        # expanded_dEdh allows us to look at various kinds of distributions which get “summed-away” in backprop.
        delta_y_broadcastable = delta_y.unsqueeze(1)
        expanded_dEdh = torch.mul(delta_y_broadcastable, w2[:-1, :])

        # Experiment
        if (t == 0 or t == (epochs - 1)) and b == (num_of_batches - 1):
            hist_min = -0.05#torch.min(expanded_dEdh[:, 1, 1])*1.5
            hist_max = 0.05 #torch.max(expanded_dEdh[:, 1, 1])*1.5
            ax1.hist(torch.mean(expanded_dEdh, dim=0)[1, :].numpy(), bins=51, range=(hist_min, hist_max), histtype='step')
            ax2.hist(torch.mean(expanded_dEdh, dim=0)[2, :].numpy(), bins=51, range=(hist_min, hist_max), histtype='step')
            ax3.hist(torch.mean(expanded_dEdh, dim=0)[3, :].numpy(), bins=51, range=(hist_min, hist_max), histtype='step')
            #plt.show(block=(b == num_of_batches - 1))
            #plt.draw()
            hist_test = torch.histc(expanded_dEdh[:, 1, 1], bins=50, min= torch.min(expanded_dEdh), max=torch.max(expanded_dEdh), out=None)

        weight_scaling = 1
        if False:
            # This block of code is for tweaking backprop. with information derived from distributions.
            expanded_dEdh_sum = torch.sum(expanded_dEdh)
            dEdh_var = torch.var(expanded_dEdh, dim=0)
            dEdh_mean = torch.mean(expanded_dEdh, dim=0)
            dEdh_var_normalized = torch.div(dEdh_var, dEdh_mean)
            #softmax = torch.nn.Softmax()
            #dEdh_var = softmax(dEdh_var).data
            #dEdh_var_sumed = torch.sum(dEdh_var, dim=1).unsqueeze(1)
            #dEdh_var_normalized = torch.div(dEdh_var, dEdh_var_sumed).unsqueeze(0)
            expanded_dEdh_adjusted = torch.div(expanded_dEdh, dEdh_var)
            expanded_dEdh_adjusted_sum = torch.sum(expanded_dEdh_adjusted)
            weight_scaling = expanded_dEdh_sum / expanded_dEdh_adjusted_sum
            # learning_rate = 0.01

        dEdh = torch.sum(expanded_dEdh, dim=2).mul(weight_scaling)
        #old_dEdh = delta_y.mm(w2[:-1, :].t())
        delta_h = torch.mul(dEdh,  dSigmoid(h))
        grad_w1 = x_batches[b].t().mm(delta_h)
        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

print('training done')


# Starting test

# picking test data
# removing samples that are present in training set
data_training_removed = np.delete(mnist.data, train_idx, 0)
target_training_removed = np.delete(mnist.target, train_idx, 0)
print(data_training_removed.shape)

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
print('test accuracy: ', accuracy)

plt.show()
