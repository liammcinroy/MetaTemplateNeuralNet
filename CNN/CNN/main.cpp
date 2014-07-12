#include <iostream>

#include "ConvolutionalNeuralNetwork.h"
#include "Layer.h"
#include "Matrix.h"

int max(int a, int b)
{
	return (a >= b) ? a : b;
}

matrix<float> convolve(matrix<float> input_matrix, matrix<float> kernal)
{
	int M = (kernal.cols - 1) / 2;
	int N = (kernal.rows - 1) / 2;
	matrix<float> result(input_matrix.cols - (2 * M), input_matrix.rows - (2 * N), 1);

	for (int k = 0; k < input_matrix.dims; ++k)
	{
		matrix<float> current(input_matrix.cols - (2 * M), input_matrix.rows - (2 * N), 1);
		//apply to every pixel
		for (int i = M; i < input_matrix.cols - M; ++i)
		{
			for (int j = N; j < input_matrix.rows - N; ++j)
			{
				//find sum
				float sum = 0.0f;
				for (int m = -M; m <= M; ++m)
					for (int n = -N; n <= N; ++n)
						sum += (input_matrix.at(i - m, j - n, k) * kernal.at(M + m, N + n, 0));
				current.set(i - M, j - N, 0, sum);
			}
		}

		//add channels
		for (int i = 0; i < result.cols; ++i)
			for (int j = 0; j < result.rows; ++j)
				result.set(i, j, 0, result.at(i, j, 0) + current.at(i, j, 0));
	}
	return result;
}

matrix<float> deconvolve(matrix<float> input_matrix, matrix<float> kernal)
{
	int N = (kernal.cols - 1) / 2;
	int M = (kernal.rows - 1) / 2;
	matrix<std::string> map(kernal.cols, kernal.rows, 1);
	matrix<std::string> codes(input_matrix.cols + (2 * N), input_matrix.rows + (2 * M), 1);

	int symmetry = 0;
	if (kernal.at(1, 0, 0) == -kernal.at(kernal.cols - 1, kernal.rows - 2, 0))//bottom left to right diagonal
		symmetry = 1;
	else if (kernal.at(1, 0, 0) == -kernal.at(0, kernal.rows - 2, 0))//bottom right to left diagonal
		symmetry = 2;
	else if (kernal.at(1, 0, 0) == -kernal.at(1, kernal.rows - 1, 0))//across
		symmetry = 3;
	else if (kernal.at(1, 0, 0) == -kernal.at(kernal.cols - 2, 0, 0))//up and down
		symmetry = 4;

	bool pos_top = (kernal.at(0, 0, 0) != 0) ? (kernal.at(0, 0, 0) > 0) : ((kernal.at(1, 0, 0) != 0) ? (kernal.at(1, 0, 0) > 0) :
		((kernal.at(0, 1, 0) != 0) ? (kernal.at(0, 1, 0) > 0) : ((kernal.at(1, 1, 0) != 0) ? (kernal.at(1, 1, 0) > 0) : false)));
	int times_symmetrical = 1;
	int unknown_constants = 1;

	if (symmetry != 4 && symmetry != 0)
	{
		for (int i = 0; i < kernal.cols; ++i)
		{
			for (int j = 0; j < kernal.rows; ++j)
			{
				if (kernal.at(i, j, 0) == 0)
				{
					map.set(i, j, 0, "U" + std::to_string(unknown_constants));
					++unknown_constants;
					break;
				}

				else
				{
					switch (symmetry)
					{
					case 1:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + std::to_string(times_symmetrical));
							map.set(j, kernal.rows - 1 - i, 0, "S" + std::to_string(-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + std::to_string(-times_symmetrical));
							map.set(j, kernal.rows - 1 - i, 0, "S" + std::to_string(times_symmetrical));
							++times_symmetrical;
						}
						break;
					case 2:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + std::to_string(times_symmetrical));
							map.set(j, i, 0, "S" + std::to_string(-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + std::to_string(-times_symmetrical));
							map.set(j, i, 0, "S" + std::to_string(times_symmetrical));
							++times_symmetrical;
						}
						break;
					case 3:
						if (pos_top)
						{
							map.set(i, j, 0, "S" + std::to_string(times_symmetrical));
							map.set(i, kernal.rows - 1 - j, 0, "S" + std::to_string(-times_symmetrical));
							++times_symmetrical;
						}

						else
						{
							map.set(i, j, 0, "S" + std::to_string(-times_symmetrical));
							map.set(i, kernal.rows - 1 - j, 0, "S" + std::to_string(times_symmetrical));
							++times_symmetrical;
						}
						break;
					}
				}
			}
		}
	}

	else if (symmetry == 4)
	{
		for (int j = 0; j < kernal.rows; ++j)
		{
			for (int i = 0; i < kernal.cols; ++i)
			{
				if (kernal.at(i, j, 0) == 0)
				{
					map.set(i, j, 0, "U" + std::to_string(unknown_constants));
					++unknown_constants;
					break;
				}

				else
				{
					if (pos_top)
					{
						map.set(i, j, 0, "S" + std::to_string(times_symmetrical));
						map.set(kernal.cols - 1 - i, j, 0, "S" + std::to_string(-times_symmetrical));
						++times_symmetrical;
					}

					else
					{
						map.set(i, j, 0, "S" + std::to_string(-times_symmetrical));
						map.set(kernal.cols - 1 - i, j, 0, "S" + std::to_string(times_symmetrical));
						++times_symmetrical;
					}
				}
			}
		}
	}

	//find all non perfectly canceled cells
	for (int i = N; i < codes.cols - N; ++i)
	{
		for (int j = M; j < codes.rows - M; ++j)
		{
			if (input_matrix.at(i - N, j - M, 0) > 0)
			{
				float difference_multiplier = input_matrix.at(i - N, j - M, 0);
				for (int x = 0; x < map.cols; ++x)
				for (int y = 0; y < map.rows; ++y)
				if (map.at(x, y, 0).substr(0, 1) == "S")
					difference_multiplier /= abs(kernal.at(x, y, 0));
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				{
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
					{
						if (codes.at(i2, j2, 0).substr(0, 1) != "S" || codes.at(i2, j2, 0).find("*") == std::string::npos
							&& input_matrix.at(i - N, j - M, 0) != 0)
							//overwrite non multipliers
							codes.set(i2, j2, 0, map.at(i2 % kernal.cols, j2 % kernal.rows, 0)
							+ "*" + std::to_string(difference_multiplier));
						else if (codes.at(i2, j2, 0).find("*") != std::string::npos && input_matrix.at(i - N, j - M, 0) != 0)
						{
							//merge multipliers
							float current_multiplier = std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()));
							std::string new_multiplier = std::to_string((current_multiplier + difference_multiplier) / 2);
							codes.set(i2, j2, 0, codes.at(i2, j2, 0).replace(codes.at(i2, j2, 0).find("*") + 1, new_multiplier.length(), new_multiplier));
						}
					}
				}
			}

			else if (input_matrix.at(i - N, j - M, 0) < 0)
			{
				float difference_multiplier = input_matrix.at(i - N, j - M, 0);
				for (int x = 0; x < map.cols; ++x)
				for (int y = 0; y < map.rows; ++y)
				if (map.at(x, y, 0).substr(0, 2) == "S-")
					difference_multiplier /= abs(kernal.at(x, y, 0));
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				{
					for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
					{
						if (codes.at(i2, j2, 0).substr(0, 1) != "S" || codes.at(i2, j2, 0).find("*") == std::string::npos
							&& input_matrix.at(i - N, j - M, 0) != 0)
							//overwrite non modifiers
							codes.set(i2, j2, 0, map.at(i2 % kernal.cols, j2 % kernal.rows, 0)
							+ "*" + std::to_string(difference_multiplier));
						else if (codes.at(i2, j2, 0).find("*") != std::string::npos && input_matrix.at(i - N, j - M, 0) != 0)
						{
							//merge multipliers
							float current_multiplier = std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()));
							std::string new_multiplier = std::to_string((current_multiplier + difference_multiplier) / 2);
							codes.set(i2, j2, 0, codes.at(i2, j2, 0).replace(codes.at(i2, j2, 0).find("*") + 1, new_multiplier.length(), new_multiplier));
						}
					}
				}
			}
		}
	}

	//find all perfectly canceled cells
	for (int i = N; i < codes.cols - N; ++i)
	{
		for (int j = M; j < codes.rows - M; ++j)
		{
			if (input_matrix.at(i - N, j - M, 0) == 0)
			{
				//map for multipliers for each
				std::map<std::string, float> multipliers;
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
				if (kernal.at(kernal.cols - (i2 - kernal.cols), kernal.rows - (j2 - kernal.rows), 0) != 0)
				if (multipliers.find(codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1)) == multipliers.end())
					multipliers.insert(std::pair<std::string, float>(codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1),
					std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, codes.at(i2, j2, 0).length()))));
				//merge symmetric
				for (int i2 = 0; i2 < times_symmetrical; ++i2)
				{
					float new_multiplier = (multipliers["S" + i2] + multipliers["S" + (-i2)]) / 2;
					multipliers["S" + i2] = new_multiplier;
					multipliers["S" + (-i2)] = new_multiplier;
				}
				for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
				for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
				if (kernal.at(kernal.cols + (i2 - kernal.cols), kernal.rows + (j2 - kernal.rows), 0) != 0)
					codes.set(i2, j2, 0, codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1) + "*"
					+ std::to_string(multipliers[codes.at(i2, j2, 0).substr(0, codes.at(i2, j2, 0).find("*") - 1)]));
			}
		}
	}

	matrix<float> result(codes.cols, codes.rows, 1);

	//convert to vectors
	std::vector<float> kernal_vector;
	for (int i = 0; i < kernal.cols; ++i)
	for (int j = 0; j < kernal.rows; ++j)
	if (kernal.at(i, j, 0) != 0)
		kernal_vector.push_back(kernal.at(i, j, 0));

	for (int i = N; i < codes.cols - N; ++i)
	{
		for (int j = M; j < codes.rows - M; ++j)
		{
			float current_output = input_matrix.at(i - N, j - M, 0);

			std::vector<float> multiples_vector;
			int n = 0;
			int m = 0;
			for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
			{
				for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
				{
					if (kernal.at(n, m, 0) != 0)
						multiples_vector.push_back(std::stof(codes.at(i2, j2, 0).substr(codes.at(i2, j2, 0).find("*") + 1, 
						codes.at(i2, j2, 0).length())));
					++m;
				}
				m = 0;
				++n;
			}

			float substitue_value = current_output / multiples_vector.size();

			//substitute in
			n = 0;
			int n2 = 0;
			for (int i2 = i - N; i2 < map.cols + i - N; ++i2)
			{
				for (int j2 = j - M; j2 < map.rows + j - M; ++j2)
				{
					if (kernal.at(n, m, 0) != 0)
					{
						result.set(i2, j2, 0, substitue_value * multiples_vector[n2]);
						++n2;
					}
					++m;
				}
				m = 0;
				++n;
			}
		}
	}

	return result;
}

void print_matrix(matrix<float> input)
{
	for (int i = 0; i < input.cols; ++i)
	{
		for (int j = 0; j < input.rows; ++j)
			std::cout << input.at(i, j, 0) << " ";
		std::cout << std::endl;
	}
}

int main(int argc, const char* args[])
{
	matrix<float> initial;
	initial = { 
	{ 1, 1, 1, 1, 1 },
	{ 2, 2, 2, 2, 2 },
	{ 3, 3, 3, 3, 3 },
	{ 4, 4, 4, 4, 4 },
	{ 5, 5, 5, 5, 5 }
	};

	matrix<float> kernal;
	kernal = {
		{ -1, -1, -1 },
		{ 0, 0, 0 },
		{ 1, 1, 1 }
	};

	std::cout << "Convolved:" << std::endl;
	matrix<float> convolved = convolve(initial, kernal);
	print_matrix(convolved);

	std::cout << "\n\n" << "Deconvolved:" << std::endl;
	matrix<float> deconvolved = deconvolve(convolved, kernal);
	print_matrix(deconvolved);

	std::cout << "\n\n" << "Convolved with deconvolved:" << std::endl;
	matrix<float> reconvolved = convolve(deconvolved, kernal);
	print_matrix(reconvolved);

	char c;
	std::cin >> c;
	return 0;
}