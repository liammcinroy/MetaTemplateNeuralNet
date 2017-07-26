# MetaTemplateNeuralNet
==========================

An API for neural networks implemented in C++ with template meta-programming. Perhaps the first of its kind.

## Include
==========================

All the necessary components are in header files (yay templates!) So just add what you need from the include folder.


## What a Neural Network is
==========================

Our brains work by a large web of connected neurons, or simple binary states. These neurons are connected by synapses, which have a strength associated with them. When a neuron fires, it's signal is sent through all of it's connecting synapses to other neurons to determine their value. When we learn, our brain adjusts the strengths of the associated synapses to limit the amount of activated neurons.

A neural network is a machine learning algorithm based off of the brain. Within a network, there are layers. Each of these layers has a number of neurons, which take on floating point values, and weights, symbolic of synapses, attached to the neurons in the next layer. These networks then run in a way similar to our brains, given an input, all neurons are fed forward to the next layer by summing the value of the neurons times the weights connecting two neurons. Commonly, a bias is an addition to the network which is used as a simple shift to neurons. The bias is added to the sum of the weights times the neurons to produce the output of the neuron, which is then commonly ran through a continuous activation function, such as a sigmoid, to bound the value of the neuron as well as give the network a differentiable property.

Weights can be connected between neurons in different ways. Most common are full connectivity layers and shared weight layers. Full connectivity layers have weights going from every input neuron to every output neuron, so every neuron in the layers are connected to every neuron in the layers above. Shared weights are a way of forming similar connections between different neurons by a common weight pattern. A common implementation of this is convolutional layers.

Convolutional layers make use of mathematical convolution, an operation used to produce feature maps, or highlights from an image. Convolution is formally defined as the sum of all values in the domains of two functions which are multiplied by one another. In real life cases, this is commonly discrete, and is most easily understood in images. Image convolution involves iterating a mask over an image to produce an output, where the output pixel values are equivelant to the sum of the mask multiplied by neighboring pixels in the input when anchored at the center of the mask. This operation draws features from the image, such as edges or curves, and is associated with the way our visual cortex processes imagery.

Networks learn through different algorithms, although the two implemented here are the up-down or wake-sleep algorithm and vanilla backpropagation. Backpropagation is an algorithm which computes the derivatives of the error with respect to the weights, and adjusts the weights in order to find a minimum in the error function. This is a way of approximating the actual error signal of every neuron, so a small step size is often used to prevent divergence. The wake-sleep or up-down algorithm trains the network without knowledge of data in an encoder-decoder format. The layers in the network are fed forward, backwards, and forwards again, before a difference is calculated to adjust the weights. 

## How this API is implemented
=================================

This API is based off of template meta-programming to optimize efficiency. Therefore, much of this API is based on the assumption that a network architecture will be defined at compile time rather than runtime. This has caused most of the class to become static, therefore you may want to `typedef NeuralNet<...> Net;` in your source file for clarity. More details on accessing a `NeuralNet` can be found in it's section.

Note that because this is a template based approach, then almost all errors will be indescript compiler errors. Generally it is because a particular layer does not "connect" to the next.

## Documentation
===============================

### Macros
===============================

These macros are used to signify layer types, optimization methods, loss functions, and activation functions. They are prefixed with `MTNN_FUNC_*` for activation functions, `MTNN_LAYER_*` for layers, `MTNN_OPT_*` for optimization methods, and `MTNN_COST_*` for cost functions. Their name should explain their use. The available layers can be found below.

Available activation functions are linear (y = x), sigmoid (y = 1/(1 + exp(-x)), bipolar sigmoid (y = 2/(1 + exp(-x)) - 1), tanh (y = tanh), and rectified linear (y = max(0, x)).

Available loss functions are quadratic, cross entropy, log likelihood, and custom targets.

Available optimization methods are vanilla backprop (with momentum, l2 weight decay, etc. as desired), Adam, and Adagrad.


### `Matrix2D<T, size_t, size_t>`
===============================

This class is a simple matrix implementation, with some extra methods that can be used in situations outside of this neural network.

| Member/Method  | Type | Details |
|---------|------|---------|
| `data` | `std::vector<T>(rows * cols)` | holds the matrice's data in column major format |
| `at(size_t i, size_t j)` | `T` | returns the value of the matrix at i, j |
| `clone()` | `Matrix2D<T, rows, cols>` | creates a deep copy of the matrix |
| `rows()` | `static constexpr size_t` | returns the amount of rows |
| `cols()` | `static constexpr size_t` | returns the amount of cols |

<small>This table contains methods used only in the source code of the network</small>

<small>Can be initialized with initialization lists, so brace initializers may create some problems.</small>

### `FeatureMap<size_t, size_t, size_t, T = float>`
===============================

This class is a slightly more advanced wrapper of just a `std::vector<Matrix2D<T, r, c>(f)`, with basic initialization functions.

<small>Can be initialized with initialization lists, so brace initializers may create some problems.</small>

### Layer
===============================

There is no `Layer` class, but all of the "`*Layer`" classes are implemented similarily. Note that only members that are used will be used as this API uses implicit static initialization.

Use the `index` parameter to create different instances if using the same type of layer multiple times (eg. if using a `InputLayer` taking 1 input on multiple networks, add a distinct `index` to prevent them from modifying each other's data) 

| Member/Method | Type | Details |
|--------|------|----------|
| `feed_forwards(feature_maps_type& input, out_feature_maps_type& output, ...)` | `void` | Feeds the layer forward |
| `feed_backwards(feature_maps_type& output, out_feature_maps_type& input, ...)` | `void` | Feeds the layer backwards using generative biases (if bool is enabled) |
| `back_prop(...)` | `void` | Performs vanilla backpropagation with the specified activation method |
| `feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, ...)` | `void` | Feeds the layer forward (overloaded for batches) |
| `feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, ...)` | `void` | Feeds the layer backwards using generative or recognition weights (overloaded for batches) |
| `back_prop(...)` | `void` | Performs vanilla backpropagation with the specified activation method (overloaded for batches) |
| `wake_sleep(...)` | `void` | Performs the wake-sleep (up-down) algorithm with the specified activation method |
| `feature_maps` | `FeatureMap<>` | Holds current activations |
| `weights` | `FeatureMap<>` | Holds the weights |
| `biases` | `FeatureMap<>` | Holds the biases (if used) |
| `generative_biases` | `FeatureMap<>` | Holds the generative biases (if used) |
| `weights_momentum` | `FeatureMap<>` | Holds the weights' momentum |
| `biases_momentum` | `FeatureMap<>` | Holds the biases' momentum |
| `weights_aux_data` | `FeatureMap<>` | Holds the weights' aux_data (used for optimization methods) |
| `biases_aux_data` | `FeatureMap<>` | Holds the biases' aux_data (used for optimization methods) |
| `feature_maps_type` | `type` | the type |
| `out_feature_maps_type` | `type` | the type |
| `weights_type` | `type` | the type |
| `biases_type` | `type` | the type |
| `generative_biases_type` | `type` | the type |
| `feature_maps_vector_type` | `type` | the type |
| `out_feature_maps_vector_type` | `type` | the type |
| `weights_vector_type` | `type` | the type |
| `biases_vector_type` | `type` | the type |
| `generative_biases_vector_type` | `type` | the type |

### `PerceptronFullConnectivityLayer<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases>`
===============================

Basic fully connected perceptron layer.

### ConvolutionLayer<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding = true>`
===============================

Basic convolutional layer, masks or kernels must be square (but not odd!).

With padding, then output is same size. Otherwise output is reduced.

### `LSTMLayer<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store>`
===============================

Basic LSTM layer (uses tanh activation). STILL IN DEVELOPMENT, WON'T WORK WITH THREADS.

`max_t_store` states how many time steps to perform bptt on.

### `MaxpoolLayer<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols>`
===================================

Basic maxpooling layer. Maxpool is performed on each feature map independently.


### `SoftMaxLayer<size_t index, size_t features, size_t rows, size_t cols>`
=====================================

Basic softmax layer. This will compute derivatives for any cost function, not just log-likelihood. Softmax is performed on each feature map independently.

### `BatchNormalizationLayer<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function>`
=====================================

Basic batch normalization layer. Gamma and beta are in `weights` and `biases`.

<b>If using, then batch learning and the respective overloads must be used.</b>

### `InputLayer<size_t index, size_t features, size_t rows, size_t cols>`
=====================================

Basic input layer just to signify the beginning of the network. Required

### `OutputLayer<size_t index, size_t features, size_t rows, size_t cols>`
=====================================

Basic output layer just to signify the end of the network. Required

### `NeuralNetwork<typename... layers>`
===============================

This is the class that encapsulates all of the rest. Has all required methods. Will add support for more loss functions and optimization methods later. If you want to train the network in parallel (or keep different sets of weights for a target network, or different architecture, etc.) then create a new instance of the class. Each new instance has its own weights and gradients and is thread safe (if you use `*_thread(.)` functions). The static class is the master net and retains its own weights.

| Member/Method | Type | Details |
|--------|------|----------|
| `learning_rate` | `float` | The learning term of the network. Default value is 0.01 |
| `momentum_term` | `float` | The momentum term (proportion of learning rate when applied to momentum) of the network. Between 0 and 1. Default value is 0 |
| `dropout_probability` | `float` | The probability that a given neuron will be "dropped". Default value is .5 |
| `loss_function` | `size_t` | The loss function to be used. Default mean square |
| `optimization_method` | `size_t` | Optimization method to be used. Default backprop |
| `use_batch_learning` | `bool` | Whether you will apply gradient manually with minibatches |
| `use_dropout` | `bool` | Whether to train the network with dropout |
| `use_momentum` | `bool` | Whether to train the network with momentums. Cannot be used with Adam or Adagrad |
| `labels` | `FeatureMap<>` | The current labels |
| `input` | `FeatureMap<>` | The current input |
| `setup()` | `void` | Initializes the network to learn. Must call if learning. Must set the hyperparameters before calling |
| `apply_gradient()` | `void` | Updates weights |
| `save_data<typename path>()` | `void` | Saves the data. Check the example to see how to supply the filename |
| `load_data<typename path>()` | `void` | Loads the data (<b>Must have initialized network and filled layers first!!!</b>) |
| `set_input(FeatureMap<> input)` | `void` | Sets the current input |
| `set_labels(FeatureMap<> labels)` | `void` | Sets the current labels |
| `discriminate()` | `void` | Feeds the network forward with current input, can be specified |
| `discriminate(FeatureMapVector<> inputs)` | `void` | Feeds the network forward with the batch inputs |
| `generate(FeatureMap<> input, size_t sampling_iterations, bool use_sampling)` | `FeatureMap<>` | Generates an output for an rbm network. `use_sampling` means sample for each layer after the markov iterations on the final RBM layer |
| `pretrain()` | `void` | Pretrains the network using the wake-sleep algorithm. Assumes every layer upto the last RBM layer has been trained. |
| `train()` | `float` | Trains the network using specified optimization method. `already_fed` means that the network has already been discriminated and the algorithm does not need to get the hidden layer activations. |
| `train_batch(FeatureMapVector<> batch_inputs, FeatureMapVector<> batch_labels)` | `float` | Trains the network using specified optimization method and batch learning. `already_fed` means that the network has already been discriminated and the algorithm does not need to get the hidden layer activations. MUST BE USED IF USING BATCH NORMALIZATION |
| `discriminate_thread()` | `void` | Feeds the network forward with current input and the current initialization (or thread's) weights, can be specified. |
| `discriminate_thread(FeatureMapVector<> inputs)` | `void` | Feeds the network forward with the batch inputs and the current initialization (or thread's) weights. |
| `train_thread()` | `float` | Trains the network using specified optimization method with the current initialization (or thread's) weights.  `already_fed` means that the network has already been discriminated and the algorithm does not need to get the hidden layer activations. |
| `train_batch_thread(FeatureMapVector<> batch_inputs, FeatureMapVector<> batch_labels)` | `float` | Trains the network using specified optimization method and batch learning with the current initialization (or thread's) weights.  `already_fed` means that the network has already been discriminated and the algorithm does not need to get the hidden layer activations. MUST BE USED IF USING BATCH NORMALIZATION |
| `calculate_population_statistics(FeatureMapVector<> batch_inputs)` | `void` | Calculates the population statistics for BN networks. Do after all training with full training data. |
| `template get_layer<size_t l> | `type` | Returns the lth layer's type |
| `template loop_up_layers<template<size_t l> class loop_body, typename... Args> | `type` | Initialize one of these to perform a function specified from the initialization of a `loop_body` type on each layer with initialization arguments of type `Args...` |
| `template loop_down_layers<template<size_t l> class loop_body, typename... Args> | `type` | Initialize one of these to perform a function specified from the initialization of a `loop_body` type on each layer with initialization arguments of type `Args...` |

### `NeuralNetAnalyzer<typename Net>`

This is a singleton static class. This class helps with network analysis, such as the expected error, and finite difference backprop checking.

| Member/Method | Type | Details |
|--------|------|----------|
| `sample_size` | `static size_t` | The sample size used to calculate the expected error |
| `mean_gradient_error()` | `static std::pair<float, float>` | Uses finite differences for backprop checking, returns mean difference in ordered pair (weights, biases) |
| `proportional_gradient_error()` | `static std::pair<float, float>` | Uses finite differences for backprop checking, returns proportional difference in ordered pair (weights, biases) |
| `add_point(float value)` | `static void` | Adds a point for the running calculation of the expected error |
| `mean_error()` | `static float` | Returns the running estimate of expected error |
| `save_error(std::string path)` | `static void` | Saves all calculated expected errors |


# Usage
===============================

For an example of creating and using a network, see main.cpp in the examples folder.

There is also an example with the MNIST Database in the examples folder. The provided .nn file has ~1% error.