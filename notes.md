### Plan / To do list
* Create a simple network to learn MNIST (done)
* Make the network similar to Ng's course assignment and replicate the results. (done)
* Plot learning rate and PDF on the delta's for each neuron in a mini-batch. (done)
* Calculate stats for the delta's for each neuron. (done)
* Update the back prop. to take into account uncertainty in delta's and measure performance (i.e. learning rate ... ) (done, need to document in the report)
* Do better random initialization to get consistent convergence. (done)
* Train the network on 8 of the 10 classes, then test on the 2 classes unseen during training and measure distribution of activations (in the hidden layer?).
    * See if the distributions of activations for the unseen classes can be grouped by a simple unsupervised clustering algorithm.
* Idea: Change this to a Jupyter notebook for easier tweaking and better explanation of what's going on

### Thoughts
As the network learns, distribution of deltas  becomes narrower and centers around zero, because the network is making less mistakes.
If the adjustment favors output neurons with higher uncertainty in their delta's then neurons in the hidden layer can be seen as helping output neurons that are bad in predicting more than the ones that are good. But isn't that already built-in?

### Types of distribution to look at
In the standard mini-batch backpropogation this information is summed-away, but it could contain useful info.
1. Dist. of deltas for each weight in batch
2. Dist. of deltas coming in each neuron from neurons in the next layer.
3. Dist. of input compared to dist of output
4. Distribution of weights? to or from a neuron?
These distributions can be marginalized for classes in training data

### Questions
1. What kind of information can you get from these distributions that you can't get from loss function?
2. How can this information be leveraged?
3. Can we determine over-fit and under fit?

### Ideas for each distribution mentioned above
* For the second type of dist., we can compare distribution of deltas coming from neurons in the next layer for a batch. One idea is that if there is a strong pull on delta from two different directions (two sharp peaks one smaller than zero and one bigger than zero) we can split the neuron to two.
* For the third type of dist, a neuron can remove itself, if there is a direct relationship between a neuron in the previous layer and a neuron in the next layer. Basically a neuron can find out if it is an unnecessary middle man.
* For the second type of dist., if the distribution is centered around zero but is wide, it means the network behind it it under fitting. It is interesting because it could tell us where to add neurons.
* For the 4th type of dist, maybe we can get rid of neurons that have a very uniform distribution of input wights. I think a very uniform weight distribution means the neuron isn't doing much?
* It also be interesting to see if looking at this distributions we could tell whether the network has too many dof and is prone to over fitting. It could help us in pruning the network.
* Use the distribution of deltas during training to create certainty info in the forward prop. 

* Gather statistics about whether the sum of deltas each neuron in the layer before received, is in the same direction as the delta you(a neuron) propagated back. This will tell you whether you are like to receive useful input from that neuron. 




