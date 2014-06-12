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
discriminating, generating, or teaching. There is also a custom file format created especially for CNNs, and is described in the "File Format Documentation"
in the source code's folder. There is also a constructor that will create new networks reliably. The following covers the methods exposed in all of the classes.

===============================

###SimpleNeuron

SimpleNeuron is a class that has the basics of a neuron. These include location using the layer and index, and the value of the neuron. The following table
describes the uses of each function

####`SimpleNeuron`
===========================

| Method Name | Parameters | Function | 
| ------------|------------|----------- | 
| `SimpleNeuron` | `int layer, int index` | Constructor for SimpleNeuron | 
| `GetValue` | _none_ | Gets the current value | 
| `SetValue` | `float newValue` | Sets the current value | 
| `GetLayer` | _none_ | Gets the current layer | 
| `GetIndex` | _none_ | Gets the current index | 
-------------------------------------------------

###Synapse

Synapse is a class containing two SimpleNeurons, a parent and a child. The parent is the SimpleNeuron that the Synapse originates from, and the child is 
the destination SimpleNeuron. There are two weights, both a discriminative and a generative.

####`Synapse`
===========================

| Method Name | Parameters | Function | 
| ------------|------------|-----------| 
| `Synapse` | `SimpleNeuron parent, SimpleNeuron child` | Constructor for Synapse | 
| `GetParent` | _none_ | Gets the parent SimpleNeuron | 
| `GetParent` | _none_ | Gets the parent SimpleNeuron | 
| `GetChild` | _none_ | Gets the child SimpleNeuron | 
| `GetWeightDiscriminate` | _none_ | Gets the discriminative weight | 
| `SetWeightDiscriminate` | `float newValue` | Sets the new discriminative weight | 
| `GetWeightGenerative` | _none_ | Gets the generative weight | 
| `SetWeightGenerative` | `float newValue` | Sets the new generative weight | 
------------------------------------------------------------------------------------------

###Neuron

Neuron is a class that inherits from SimpleNeuron, and contains vectors of the Synapses that it is both a parent and a child of. It also has the 
code that makes the CNN work. The entire network is made of Neurons. This class also enables the network to fire synapses.

####`Neuron`
===========================

| Method Name | Parameters | Function | 
| ------------|------------|-----------| 
| `Neuron` | `std::vector<Synapse> parentOf, std::vector<Synapse> childOf` | Constructor for Neuron | 
| `GetValue` | _none_ | Gets the current value | 
| `SetValue` | `float newValue` | Sets the current value | 
| `GetLayer` | _none_ | Gets the current layer | 
| `GetIndex` | _none_ | Gets the current index | 
| `AddParentOfSynapse` | `SimpleNeuron child` | Adds a parent of synapse | 
| `AddChildOfSynapse` | `SimpleNeuron parent` | Adds a child of synapse | 
| `GetParentOfSynapses` | _none_ | Gets the parent of synapses | 
| `GetParentOfSynapseAt` | `int index` | Gets the parent of synapse at index | 
| `GetChildOfSynapses` | _none_ | Gets the child of synapses | 
| `GetChildOfSynapseAt` | `int index` | Gets the child of synapse at index | 
| `FireSynapse` | _none_ | Returns the value of the next neuron when discriminating | 
| `FireInverseSynapse` | _none_ | Returns the value of the next neuron when generating | 
-------------------------------------------------------------------------------------------------

###Layer

A layer has many neurons, and many methods to interact with those neurons

####`Layer`
===========================

| Method Name | Parameters | Function | 
| ------------|------------|-----------| 
| `Layer` | `std::vector<Neuron> neurons` | Constructor for Layer | 
| `GetNeurons` | _none_ | Gets the vector of Neurons | 
| `GetNeuronAt` | `int index` | Gets the Neuron at index | 
| `AddNeuron` | `Neuron neuron` | Adds a neuron to the end of the Layer | 
| `FireNeuronAt` | `int index` | Fires the neuron for discriminating at index | 
| `FireInverseNeuronAt` | `int index` | Fires the neuron for generating at index | 
----------------------------------------------------------------------------------------

###ConvolutionalNeuralNetwork

This class is the result of the earlier hierarchy. It contains methods for teaching, discriminating, generating, saving, loading, and creating a network.

####`ConvolutionalNeuralNetwork`
===========================

| Method Name | Parameters | Function | 
| ------------|------------|----------- | 
| `ConvolutionalNeuralNetwork` | `std::string path` | Constructor for network, reads data from path | 
| `ConvolutionalNeuralNetwork` | `std::vector<int> neuronCountPerLayer, std::vector<int> featureMapsPerLayer, std::vector<int> featureMapDimensions,
		std::vector<std::vector<int>> featureMapConnections, std::vector<std::vector<int>> featureMapStartIndex` | Constructor for network, creates new network with specified attributes | 
| `GetLayers` | _none_ | Gets the vector of Layers | 
| `GetLayerAt` | `int index` | Gets the Layer at index | 
| `AddLayer` | `Layer newLayer` | Adds a new Layer at the end of the network | 
| `GetInput` | _none_ | Gets the current input | 
| `SetInput` | `std::vector<std::vector<float>> input` | Sets the current input | 
| `GetOutput` | _none_ | Gets the current output | 
| `GetLearningRate` | _none_ | Gets the learning rate | 
| `SetLearningRate` | `float newRate` | Sets the learning rate | 
| `Discriminate` | _none_ | Discriminates using current input | 
| `Generate` | `Layer input` | Generates using the given input from a resulting output | 
| `LearnCurrentInput` | _none_ | Learns the current input | 
| `ReadFromFile` | `std::string path` | Clears network and sets to data from path | 
| `SaveToFile` | `std::string path` | Saves the network to the path | 
 ------------------------------------------------------------------------------------------------------------------------

 
 ##Performance
 =================================
 
 This was code is unoptimized and _very_ slow. To create a new network takes 30 seconds. 
 To read a network with 3000 neurons from a file took 1530 seconds (~25 minutes).
 To discriminate took over 2 days, and I didn't have the patience for the generation
 This code will be optimized once it is know to be working.
 
 #####Update
 
 The code created a network in 18 milliseconds, saved in about 3 seconds, Discriminated in about 5 minutes,
 and generated in only 2 seconds. This is about 2000+x faster.