#include "ilayer.h"

ILayer::~ILayer()
{
	delete[] feature_maps;
	delete[] data;
}

void ILayer::find_probability()
{
	for (int f = 0; f < feature_maps_count; ++f)
	{
		for (int i = 0; i < feature_maps[f].rows; ++i)
		{
			for (int j = 0; j < feature_maps[f].cols; ++j)
			{
				float prob = 1 / (1 + exp((float)-feature_maps[f].at(i, j)));
				feature_maps[f].at(i, j) = (rand() <= prob) ? 1 : 0;
			}
		}
	}

}

void ILayer::dropout()
{
	for (int f = 0; f < feature_maps_count; ++f)
	for (int i = 0; i < feature_maps[f].rows; ++i)
	for (int j = 0; j < feature_maps[f].cols; ++j)
	if (rand() >= .5f)
		feature_maps[f].at(i, j) = 0;
}

void ILayer::wake_sleep(ILayer &above, float &learning_rate)
{
	//find difference via gibbs sampling
	Matrix2D<int>* discriminated = (Matrix2D<int>*)malloc(out_features * sizeof(Matrix2D<int>(out_rows, out_cols)));
	discriminated = this->feed_forwards();
	Matrix2D<int>* generated = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(above.in_rows, above.in_cols)));
	generated = above.feed_backwards(data);
	Matrix2D<int>* temp_feature = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(Matrix2D<int>(above.in_rows, above.in_cols)));
	temp_feature = feature_maps;
	feature_maps = generated;
	Matrix2D<int>* reconstructed = (Matrix2D<int>*)malloc(out_features * sizeof(Matrix2D<int>(out_rows, out_cols)));
	reconstructed = this->feed_forwards();
	feature_maps = temp_feature;
	free(temp_feature);

	Matrix2D<int>* difference = (Matrix2D<int>*)malloc(out_features * sizeof(Matrix2D<int>(out_rows, out_cols)));
	for (int i = 0; i < out_features; ++i)
		difference[i] = discriminated[i] - reconstructed[i];

	//adjust weights
	for (int f = 0; f < feature_maps_count; ++f)
	for (int i = 0; i < data_count; ++i)
	for (int j = 0; j < feature_maps[f].rows; ++j)
		data[f].at(i, j) += learning_rate * (discriminated[f].at(i, 0) - reconstructed[f].at(i, 0));

	//clean up
	free(discriminated);
	free(generated);
	free(reconstructed);
	free(difference);
}

ConvolutionLayer::ConvolutionLayer(int features, int rows, int cols, int kernel_rows, int kernel_cols, int output_features, int input_features,
	int input_rows, int input_cols, int conv_stride)
{
	feature_maps_count = features;
	data_count = output_features;
	out_features = output_features;
	out_rows = rows - (kernel_rows - 1);
	out_cols = cols - (kernel_cols - 1);
	in_rows = input_rows;
	in_cols = input_cols;
	stride = conv_stride;
	type = CNN_CONVOLUTION;
	Matrix2D<int> feature_sample(rows, cols);
	feature_maps = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(feature_sample));
	for (int i = 0; i < features; ++i)
		feature_maps[i] = feature_sample;
	Matrix2D<float> data_sample(kernel_rows, kernel_cols);
	data = (Matrix2D<float>*)malloc(data_count * sizeof(data_sample));
	for (int i = 0; i < data_count; ++i)
		data[i] = data_sample;
}

ConvolutionLayer::~ConvolutionLayer()
{
	free(feature_maps);
	free(data);
}

Matrix2D<int> ConvolutionLayer::convolve(Matrix2D<int> &input, Matrix2D<float> &kernel)
{
	int N = (kernel.rows - 1) / 2;
	Matrix2D<int> output(input.rows - (2 * N), input.cols - (2 * N));

	for (int i = N; i < input.rows - N; i += stride)
	{
		for (int j = N; j < input.cols - N; j += stride)
		{
			int sum = 0;
			for (int n = N; n >= -N; --n)
			for (int m = N; m >= -N; --m)
				sum += input.at(i - n, j - m) * kernel.at(N - n, N - m);
			output.at(i - N, j - N) = sum;
		}
	}
	return output;
}

FeedForwardLayer::FeedForwardLayer(int features, int rows, int output_rows, int input_num)
{
	out_rows = output_rows;
	out_cols = 1;
	out_features = features;
	in_rows = input_num;
	in_cols = 1;
	in_features = features;
	feature_maps_count = features;
	type = CNN_FEED_FORWARD;
	Matrix2D<int> feature_sample(rows, 1);
	feature_maps = (Matrix2D<int>*)malloc(features * sizeof(feature_sample));
	for (int i = 0; i < features; ++i)
		feature_maps[i] = feature_sample;
	Matrix2D<float> data_sample(out_rows, rows);
	data = (Matrix2D<float>*)malloc(features * sizeof(data_sample));
	for (int i = 0; i < out_rows; ++i)
		data[i] = data_sample;
}

FeedForwardLayer::~FeedForwardLayer()
{
	free(feature_maps);
	free(data);
}

MaxpoolLayer::MaxpoolLayer(int features, int rows, int cols, int maxed_rows, int maxed_cols)
{
	feature_maps_count = features;
	out_rows = maxed_rows;
	out_cols = maxed_cols;
	out_features = feature_maps_count;
	in_rows = 0;
	in_cols = 0;
	in_features = 0;
	type = CNN_MAXPOOL;
	Matrix2D<int> feature_sample(rows, cols);
	feature_maps = (Matrix2D<int>*)malloc(feature_maps_count * sizeof(feature_sample));
	for (int i = 0; i < feature_maps_count; ++i)
		feature_maps[i] = feature_sample;
}

MaxpoolLayer::~MaxpoolLayer()
{
	free(feature_maps);
}

