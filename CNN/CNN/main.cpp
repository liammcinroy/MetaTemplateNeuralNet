#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

extern std::vector<std::string> split(std::string &input, const std::string &delim);

template<typename T> bool is_equal(Matrix<T>* &first, Matrix<T>* &other)
{
	if (first->rows() == other->rows() && first->cols() == other->cols())
	{
		for (int i = 0; i < first->rows(); ++i)
			for (int j = 0; j < first->cols(); ++j)
				if (first->at(i, j) != other->at(i, j))
					return false;
	}
	else
		return false;
	return true;
}

int main(int argc, char** argv)
{
	srand(time(NULL));

	//Example network for determining an Abalone's gender from height and weight (Note it is unknown whether the data have any relationship)
	//taken from http://archive.ics.uci.edu/ml/datasets/Abalone
	 /*
	NeuralNet net = NeuralNet();
	net.add_layer(new FeedForwardLayer<1, 2, 20>());
	net.add_layer(new FeedForwardLayer<1, 20, 10>());
	net.add_layer(new FeedForwardLayer<1, 10, 5>());
	net.add_layer(new FeedForwardLayer<1, 5, 3>());
	net.add_layer(new FeedForwardLayer<1, 3, 3>());
	net.add_layer(new OutputLayer<1, 3, 1>());

	net.load_data("C://abalone_data.cnn");
	int t = clock();
	net.use_dropout = false;
	net.learning_rate = 0.05f;
	net.binary_net = true;

	std::ifstream data("C://abalone_data.txt");
	t = clock();
	int i = 0;
	std::string current;
	 */

	//pretraining
	 /*
	std::cout << "Pretraining network..." << std::endl;
	while (std::getline(data, current) && i < 3000)
	{
		std::vector<std::string> current_data = split(current, ",");
		float height = std::stof(current_data[3]);
		float weight = std::stof(current_data[4]);

		std::vector<Matrix<float>*> input;
		input.push_back(new Matrix2D<float, 2, 1>());
		input[0]->at(0, 0) = height;
		input[0]->at(1, 0) = weight;

		net.set_input(input);

		std::vector<Matrix<float>*> labels;
		labels.push_back(new Matrix2D<float, 3, 1>());
		labels[0]->at(0, 0) = (current_data[0] == "M" ? 1 : 0);
		labels[0]->at(1, 0) = (current_data[0] == "F" ? 1 : 0);
		labels[0]->at(2, 0) = (current_data[0] == "I" ? 1 : 0);

		net.set_labels(labels);

		net.pretrain();

		if (i % 25 == 0)
			net.save_data("C://abalone_data.cnn");
		++i;
	}

	std::cout << "\n\tNetwork pretrained with " << i << " inputs; Total elapsed time: " << clock() - t << "\n" << std::endl;
	 */

	//backprop
	 /*
	std::cout << "Training network..." << std::endl;
	data = std::ifstream("C://abalone_data.txt");
	t = clock();
	i = 0;
	while (std::getline(data, current) && i < 3000)
	{
		std::vector<std::string> current_data = split(current, ",");
		float height = std::stof(current_data[3]);
		float weight = std::stof(current_data[4]);

		std::vector<Matrix<float>*> input;
		input.push_back(new Matrix2D<float, 2, 1>());
		input[0]->at(0, 0) = height;
		input[0]->at(1, 0) = weight;

		net.set_input(input);

		std::vector<Matrix<float>*> labels;
		labels.push_back(new Matrix2D<float, 3, 1>());
		labels[0]->at(0, 0) = (current_data[0] == "M" ? 1 : 0);
		labels[0]->at(1, 0) = (current_data[0] == "F" ? 1 : 0);
		labels[0]->at(2, 0) = (current_data[0] == "I" ? 1 : 0);

		net.set_labels(labels);

		net.train(3);

		if (i % 25 == 0)
			net.save_data("C://abalone_data.cnn");
		++i;
	}

	std::cout << "\n\tNetwork trained with " << i << " inputs; Total elapsed time: " << clock() - t << "\n" << std::endl;
	
	net.save_data("C://abalone_data.cnn");
	 */

	//testing 
	 /*
	data = std::ifstream("C://abalone_data.txt");
	t = clock();
	i = 0;
	int correct = 0;
	//std::string current;
	std::cout << "Testing network..." << std::endl;
	while (std::getline(data, current) && i < 3000)
	{
		std::vector<std::string> current_data = split(current, ",");
		float height = std::stof(current_data[3]);
		float weight = std::stof(current_data[4]);

		std::vector<Matrix<float>*> input;
		input.push_back(new Matrix2D<float, 2, 1>());
		input[0]->at(0, 0) = height;
		input[0]->at(1, 0) = weight;

		net.set_input(input);

		std::vector<Matrix<float>*> labels;
		labels.push_back(new Matrix2D<float, 3, 1>());
		labels[0]->at(0, 0) = (current_data[0] == "M" ? 1 : 0);
		labels[0]->at(1, 0) = (current_data[0] == "F" ? 1 : 0);
		labels[0]->at(2, 0) = (current_data[0] == "I" ? 1 : 0);

		ILayer* results = net.discriminate();

		if (results->feature_maps[0]->is_equal(labels[0]))
			++correct;
		++i;
	}
	std::cout << "\n\tTesting finished. " << (100.0f * correct) / i << "% were correct out of " << i << " trials.\n\tTotal elapsed time: "
		<< clock() - t << std::endl;
	 */


	//Test network for finding average value of a matrix
	/*
	NeuralNet net = NeuralNet();
	net.add_layer(new MaxpoolLayer<1, 3, 3, 3, 3>());
	net.add_layer(new ConvolutionLayer<1, 3, 3, 3, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	net.load_data("C://avg_data.cnn");
	net.binary_net = false;
	net.learning_rate = 0.005f;
	net.use_dropout = false;
	*/

	//pretraining
	 /*
	std::cout << "Pretraining network..." << std::endl;
	for (int k = 0; k < 10000; ++k)
	{
		std::vector<Matrix<float>*> input = { new Matrix2D<float, 3, 3>(0, 25) };
		net.set_input(input);

		float average = 0.0f;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				average += (input[0]->at(i, j) / 9.0f);
		std::vector<Matrix<float>*> label = { new Matrix2D<float, 1, 1>({ average }) };
		net.set_labels(label);
		
		net.pretrain();

		if (k % 50 == 0)
			net.save_data("C://avg_data.cnn");
	}
	std::cout << "\n\tNetwork pretrained\n" << std::endl;
	 */

	//backproping
	 /*
	std::cout << "Training network..." << std::endl;
	for (int k = 0; k < 10000; ++k)
	{
		std::vector<Matrix<float>*> input = { new Matrix2D<float, 3, 3>(0, 25) };
		net.set_input(input);

		float average = 0.0f;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				average += (input[0]->at(i, j) / 9);
		std::vector<Matrix<float>*> label = { new Matrix2D<float, 1, 1>({ average }) };
		net.set_labels(label);

		net.train(3);

		if (k % 50 == 0)
			net.save_data("C://avg_data.cnn");
	}
	std::cout << "\n\tNetwork trained\n" << std::endl;
	 */

	//testing
	 /*
	std::cout << "Testing network..." << std::endl;
	int correct = 0;
	for (int k = 0; k < 10000; ++k)
	{
		std::vector<Matrix<float>*> input = { new Matrix2D<float, 3, 3>(0, 25) };
		net.set_input(input);

		float average = 0.0f;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				average += (input[0]->at(i, j) / 9);
		
		float output = net.discriminate()->feature_maps[0]->at(0, 0);

		if (abs(output - average) < 1.0f)
			++correct;
	}
	std::cout << "\n\tTesting finished. " << (100.0f * correct) / 10000 << "% were correct out of " << 10000 << " trials." << std::endl;
	 */

	NeuralNet net = NeuralNet();
	net.add_layer(new MaxpoolLayer<1, 1, 1, 1, 1>());
	net.add_layer(new ConvolutionLayer<1, 1, 1, 1, 1>());
	net.add_layer(new OutputLayer<1, 1, 1>());

	net.learning_rate = 0.05f;
	net.use_dropout = false;
	net.binary_net = false;

	std::vector<Matrix<float>*> input = { new Matrix2D<float, 1, 1>({ 1 }) };
	std::vector<Matrix<float>*> labels = { new Matrix2D<float, 1, 1>({ 3 }) };

	net.set_input(input);
	net.set_labels(labels);

	for (int i = 0; i < 100; ++i)
		net.train(3);

	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}