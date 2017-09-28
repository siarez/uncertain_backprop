### Why
A lot of interesting information gets “summed away“ during mini-batch backpropogation. 
I'm curious to learn what this information can tell us.

### What
The purpose of this project is to look at distributions of gradients and weight during mini-batch, before summations.
These distributions can be marginalized in various ways to answer interesting questions.
I used these uncertainty measures to alter the back prop. algorithm, and much more. 

### How
I implemented a quick and dirty neural network for the good ol' MNIST with PyTorch (which is quickly becoming my favorite library for prototyping)
There is a slight tweak to this backprop. calculation which allows you to look at distributions of deltas that normally get reduce-summed in `matmul`s.

A blog post is in the works to explain the results of this project.