#include <iostream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

extern std::vector<std::string> split(std::string &input, const std::string &delim);

int main(int argc, char** argv)
{
	//Example network for determining an Abalone's gender from height and weight
	//taken from http://archive.ics.uci.edu/ml/datasets/Abalone
	NeuralNet net = NeuralNet();
	net.add_layer(new FeedForwardLayer<1, 2, 5>());
	net.add_layer(new FeedForwardLayer<1, 5, 3>());
	net.add_layer(new FeedForwardLayer<1, 3, 3>());
	net.add_layer(new OutputLayer<1, 3, 1>());

	net.load_data("C://abalone_data.cnn");
	int t = clock();
	net.use_dropout = false;
	net.learning_rate = 0.005f;

	std::ifstream data("C://abalone_data.txt");

	//training
	 /*
	std::cout << "\n\tPretraining network..." << std::endl;
	t = clock();
	int i = 0;
	std::string current;
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
		{
			std::cout << "\n25 epochs completed. Saving data..." << std::endl;
			net.save_data("C://abalone_data.cnn");
			std::cout << "Data Saved." << std::endl;
		}
		++i;
	}

	std::cout << "\n\tNetwork pretrained, total elapsed time: " << clock() - t << std::endl;

	std::cout << "\n\tTraining network..." << std::endl;
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

		net.train();

		if (i % 25 == 0)
		{
			std::cout << "\n25 epochs completed. Saving data..." << std::endl;
			net.save_data("C://abalone_data.cnn");
			std::cout << "Data Saved." << std::endl;
		}
		++i;
	}

	std::cout << "\n\tNetwork trained with " << i << "inputs; total elapsed time: " << clock() - t << std::endl;
	
	net.save_data("C://abalone_data.cnn");
	*/


	//testing 
	///*
	int i = 0;
	int correct = 0;
	std::string current;
	while (std::getline(data, current) && i < 1000)
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

		bool equal = (net.discriminate()->feature_maps[0] == labels[0]);
		if (equal)
		{
			std::cout << "Test case number: " << i << " was successful" << std::endl;
			++correct;
		}
		else
			std::cout << "Test case number: " << i << " was unsuccessful." << std::endl;
		++i;
	}
	std::cout << "\n\tTesting finished. " << (100.0f * correct) / i << "% were correct out of " << i << " trials." << std::endl;
	//*/

	std::cout << "\n\nPress enter to exit" << std::endl;
	std::cin.get();
	return 0;
}