#ConvolutionalNeuralNetwork
==========================

An API for a convolutional neural network implemented in C++ with the intent to increase and assist research on architectures of neural nets through

##Static Library
==========================

The build and .h files for referencing as an external static library can be found in the Builds folder.


##What a Convolutional Neural Network is
==========================

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
In each layer, the network first discriminates, then generates, then discriminates again, forming what is known as a Markov Chain. Then, alternating 
Gibbs sampling is used to then find the difference between the first discriminated layer and the layer that was discriminated after "reconstruction." 
These values are then multiplied by a small value, or the learning rate of the network. This is done for each layer going up through the network, creating
a recursion learning based off of the adjusted weights in the previous layer. This entire process is often called "pretraining" as it is less accurate than
traditional methods (such as backpropagation) but helps learn the correct neighborhood to then fine-tune the network in.

The next process in learning, backpropagation, is found using the error of the entire network. The derivative of this function is found with respect to the 
weights, so that each weight can find the role it had in the error of the network. These derivatives are then used to find the minimum of the error function. 
An issue with this algorithm is that the network can become stuck in a local minimum instead of finding the global minimum of the network.

##How this API is implemented
=================================

This API has a neural network premade, with code to discriminate, generate, and teach. Note that a new input must be set for each iteration of
discriminating, generating, or teaching. There is also a custom file format created especially for CNNs, consisting only of the data for each layer's 
weights.

###IMatrix
===============================

This class is merely a contain for `Matrix2D<T, int, int>` so that matrix sizes unknown at compile time can be computed at runtime.

###Matrix2D<T, int, int>
===============================

This class is a simple matrix implementation, with some extra methods that can be used in situations outside of this neural network.

| Member  | Type | Details |
|---------|------|---------|
| `data` | `std::array<T, rows * cols>` | holds the matrice's data in column major format |
| `at(int i, int j)` | `T` | returns the value of the matrix at i, j |
| `clone()` | `Matrix2D<T, rows, cols>` | creates a deep copy of the matrix |
| `rows()` | `int` | returns the amount of rows |
| `cols()` | `int` | returns the amount of cols |

<small>This table contains methods used only in the source code of the network</small>

###ILayer
===============================

This is the interface for all of the various layer types used in the network.

| Member | Type | Details |
|--------|------|----------|
| `feature_maps` | `std::vector<IMatrix<float>*>` | Holds the data of the network |
| `recognition_weights` | `std::vector<IMatrix<float>*>` | The feed forwards weights |
| `generation_weights` | `std::vector<IMatrix<float>*>` | The feed backwards weights |
| `feed_forwards(std::vector<IMatrix<float>*> &output)` | `virtual void` | Feeds the layer forward |
| `feed_forwards_prob(std::vector<IMatrix<float>*> &output)` | `virtual void` | Feed the layer forward using logistic activation function |
| `feed_backwards(std::vector<IMatrix<float>*> input, bool use_g_weights)` | `virtual std::vector<IMatrix<float>*>` | Feeds the layer backwards using generative or recognition weights |
| `feed_backwards_prob(std::vector<IMatrix<float>*> input, bool use_g_weights)` | `virtual std::vector<IMatrix<float>*>` | Feeds the layer backwards using generative or recognition weights and the logistic activation function |
| `dropout()` | `void` | Sets half of the neurons to 0 to prevent overfitting |
| `wake_sleep(bool binary_net)` | `void` | Performs the wake-sleep algorithm with or without the logistic activation function |


###PerceptronLayer<int features, int rows, int cols, int out_rows, int out_cols, int out_features>
===============================

Basic perceptron layer. Interprets architecture as a single dimension array.

Overloaded functions

| Function | Difference |
|----------|-------------|
| `feed_forwards` | Uses standard sums for feeding forwards |
| `feed_backwards` | Uses standard sums for feeding backwards |

###ConvolutionLayer<int features, int rows, int cols, int recognition_data_size, int stride, int out_features>
===============================

Basic convolutional layer, masks or kernels must be square and odd.

Overloaded functions

| Function | Difference |
|----------|-------------|
| `feed_forwards` | Uses convolution for feeding forwards |
| `feed_backwards` | Uses convolution for feeding backwards |

###MaxpoolLayer<int features, int rows, int cols, int out_rows, int out_cols>
===================================

Basic maxpooling layer.

Overloaded functions

| Function | Difference |
|----------|-------------|
| `feed_forwards` | Uses maxpooling for feeding forwards |
| `feed_backwards` | N/A |

###SoftMaxLayer<int features, int rows, int cols>
=====================================

Basic softmax layer. Note that when `PerceptronLayer`s feed into this, if the network is binary, those layers will not have binary outputs.

Overloaded functions

| Function | Difference |
|----------|-------------|
| `feed_forwards` | N/A |
| `feed_backwards` | N/A |

###OutputLayer<int features, int rows, int cols>
=====================================

Basic output layer just to signify the end of the network.

Overloaded functions

| Function | Difference |
|----------|-------------|
| `feed_forwards` | N/A |
| `feed_backwards` | N/A |

###`NeuralNetwork`
===============================

This is the class that encapsulates all of the rest. Has all required methods.

| Member | Type | Details |
|--------|------|----------|
| `layers` | `std::vector<ILayer*>` | All of the network's layers |
| `labels` | `std::vector<IMatrix<float>*>` | The current labels |
| `input` | `std::vector<IMatrix<float>*>` | The current input |
| `use_dropout` | `bool` | Whether to train the network with dropout |
| `binary_net` | `bool` | Whether to use the logistic activation function |
| `add_layer(ILayer* layer)` | `void` | Adds another layer to the network |
| `save_data(std::string path)` | `void` | Saves the data |
| `load_data(std::string path)` | `void` | Loads the data (<b>Must have initialized network and filled layers first!!!</b>) |
| `set_input(std::vector<IMatrix<float>*> input)` | `void` | Sets the current input |
| `set_labels(std::vector<IMatrix<float>*> labels)` | `void` | Sets the current labels |
| `discriminate()` | `ILayer*` | Feeds the network forward |
| `pretrain()` | `void` | Pretrains the network using the wake-sleep algorithm |
| `train(int epochs)` | `void` | Trains the network using backpropogation |

#Usage
===============================

For an example of creating and using a network, see main.cpp.
