#ConvolutionalNeuralNetwork
==========================

An API for a convolutional neural network implemented in C++

==========================

##What a Convolutional Neural Network is

A convolutional neural network is made up of many layers of neurons, or nodes, that hold a value. 
There are also many synapses connecting each neuron, like a link between the nodes. This model is based
off of how our brains work. In a neural network, a synapse is fired and then determines the value of the 
neuron it is connected to. Each synapse also has a weight and a bias that go into calculating the value of
the next neuron. This is similar to how our brains' synapses grow stronger with use, which is how we remember
events or information. 

Each layer in a convolutional neural network is made up of feature maps, or many neurons that share similar synapses. 
In these feature maps, the output for each neuron is dependent on the neurons in the previous layer, and their weights.
This helps segment the data into so called feature points. Many feature maps make up a layer, which all make up networks.

When the network learns, or adjusts the weights based off of labeled data, it uses a process of both discriminating and generating data.
The input layer is set, and the network will fire all the synapses up until the final output layer. This is when the network discriminates data.
 That output will then be used to go in the reverse direction, which will generate an "input". This is called the generative process.
The combination of these two processes help teach the network the proper weights needed to generate the same input as was initially given.
This is done on a layer by layer scale, and compares the data, and subtracts the differences to find the amount to increment the discriminative
weight by, while the generative weight will be decremented by the same amount. This teaches the network much faster than previous methods, such as
back-propagation. Sometimes back-propagation will be used to fine tune the weights, but is not always needed.

==============================

##How this API is implemented

This API has a neural network premade, with code to discriminate, generate, and teach. Note that a new input must be set for each iteration of
discriminating, generating, or teaching. There will be more documentation later for specifics about the methods and classes