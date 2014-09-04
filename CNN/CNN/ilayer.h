#pragma once

#include "imatrix.h"

template<class T> struct ILayer
{
public:
	~ILayer<T>()
	{
		delete[] feature_maps;
		delete[] data;
	}
	virtual Matrix2D<T>* feed_forwards();
	virtual Matrix2D<T>* feed_backwards(const Matrix2D<T>* &weights);
	void find_probability()
	{
		for (int f = 0; f < feature_maps_count; ++f)
			for (int i = 0; i < feature_maps[f].rows; ++i)
				for (int j = 0; j < feature_maps[f].cols; ++j)
					feature_maps[f].at(i, j) = 1 / (1 + exp(-feature_maps[f].at(i, j)));
	}
	void dropout()
	{
		for (int f = 0; f < feature_maps_count; ++f)
			for (int i = 0; i < feature_maps[f].rows; ++i)
				for (int j = 0; j < feature_maps[f].cols; ++j)
					if (rand() >= .5f)
						feature_maps[f].at(i, j) = 0;
	}
	void wake_sleep(const ILayer &above, const float &learning_rate)
	{
		//find difference via gibbs sampling
		IMatrix<T>[out_feature] discriminated = this->feed_forwards();;
		IMatrix<T>[feature_maps_count] generated = above.feed_backwards();
		IMatrix<T>[feature_maps_count] temp_feature = feature_maps;
		feature_maps = generated;
		IMatrix<T>[out_feature] reconstructed = this->feed_forwards();
		feature_maps = temp_feature;
		delete[] temp_feature;

		IMatrix<T>[out_feature] difference;
		for (int i = 0; i < out_feature; ++i)
			difference[i] = discriminated[i] - reconstructed[i];

		//clean up
		delete[] discriminated;
		delete[] generated;
		delete[] reconstructed;

		//adjust weights
		for (int f = 0; f < feature_maps_count; ++f)
			for (int i = 0; i < data_count; ++i)
				for (int j = 0; j < feature_maps[f].rows; ++j)
					data[f].at(i, j) += learning_rate * (discriminated[f].at(i, 0) - reconstructed[f].at(i, 0));
	}
	void back_propogate();//TODO: investigate backpropogation
	int in_rows;
	int in_cols;
	int in_features;
	int out_rows;
	int out_cols;
	int out_features;
	int feature_maps_count;
	int data_count;
	Matrix2D<T>* feature_maps;
	Matrix2D<float>* data;
};

template<class T> class ConvolutionLayer : public ILayer<T>
{
public:
	ConvolutionLayer<T>(int features, int rows, int cols, int kernel_rows, int kernel_cols, int output_features, int input_features,
		int input_rows, int input_cols, int conv_stride = 1)
	{
		feature_maps_count = features;
		data_count = output_features;
		out_features = output_features;
		in_rows = input_rows;
		in_cols = input_cols;
		stride = conv_stride;
		Matrix2D<T> feature_sample(rows, cols);
		feature_maps = new[features * sizeof(feature_sample)];
		for (int i = 0; i < features; ++i)
			feature_maps[i] = feature_sample;
		Matrix2D<float> data_sample(kernel_rows, kernel_cols);
		data = new[data_count * sizeof(data_sample)];
		for (int i = 0; i < data_count; ++i)
			data[i] = data_sample;
	}
	~ConvolutionLayer<T>()
	{
		delete[] feature_maps;
		delete[] data;
	}
	virtual Matrix2D<T>* feed_forwards()
	{
		Matrix2D<T>[data_count] output;
		for (int i = 0; i < data_count; ++i)
		{
			Matrix2D<T> total = convolve(feature_maps[0], data[i]);
			for (int f = 1; f < feature_maps_count; ++f)
				total = total + convolve(feature_maps[f], data[i]);
			output[i] = total;
		}
		return output;
	}
	virtual Matrix2D<T>* feed_backwards(const Matrix2D<T>* &weights)
	{
		Matrix2D<T>[feature_maps_count] output;
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<T> current(in_rows, in_cols);
			for (int i = 0; i < feature_maps[f].rows; ++i)
				for (int j = 0; j < feature_maps[f].cols; ++j)
					for (int n = 0; n < weights[f].rows; ++n)
						for (int m = 0; m < weights[f].cols; ++m)
							current.at(i + n, j + m) += feature_maps[f].at(i, j) * weights[f].at(n, m);
			output[f] = current;
		}
		return output;
	}
private:
	int stride;
	Matrix2D<T> convolve(const Matrix2D<T> &input, const Matrix2D<T> &kernel)
	{
		int N = (kernel.rows - 1) / 2;
		Matrix2D<T> output(input.rows - (2 * N), input.cols - (2 * N));

		for (int i = N; i < input.rows - N; i += stride)
		{
			for (int j = N; j < input.cols - N; j += stride)
			{
				T sum;
				for (int n = N; n >= -N; --n)
					for (int m = N; m >= -N; --m)
						sum += input.at(i - n, j - m) * kernel.at(N - n, N - m);
				output.at(i - N, j - N) = sum;
			}
		}
		return output;
	}
};

template<class T> class FeedForwardLayer : public ILayer<T>
{
	FeedForwardLayer<T>(int features, int rows, int output_rows, int input_num)
	{
		out_rows = ouput_rows;
		out_cols = 1;
		out_features = features;
		in_rows = input_rows;
		in_cols = 1;
		in_features = features;
		feature_maps_count = features;
		Matrix2D<T> feature_sample(rows, 1)
			feature_maps = new[features * sizeof(feature_sample)];
		for (int i = 0; i < features; ++i)
			features_maps[i] = feature_sample;
		Matrix2D<float> data_sample(out_rows, rows);
		data = new[out_rows * sizeof(data_sample)];
		for (int i = 0; i < out_rows; ++i)
			data[i] = data_sample;
	}
	~FeedForwardLayer<T>()
	{
		delete[] feature_maps;
		delete[] data;
	}
	virtual Matrix2D<T>* feed_forwards()
	{
		Matrix2D<T>[feature_maps_count] output;
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<T> current(out_rows, 1);
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < feature_maps[f].rows; ++j)
					current.at(i, 0) += (feature_maps[f].at(j, 0) * data[f].at(i, j));
			output[f] = current;
		}
		return output;
	}
	virtual Matrix2D<T>* feed_backwards(const Matrix2D<T>* &weights)
	{
		Matrix2D<T>[feature_maps_count] output;
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<T> current(in_rows, 1);
			for (int i = 0; i < in_rows; ++i)
				for (int j = 0; j < feature_maps[f].rows; ++j)
					current.at(i, 0) += weights[f].at(j, i) * feature_maps[f].at(j, 0);
			output[f] = current;
		}
		return output;
	}
};

template<class T> class MaxpoolLayer : public ILayer<T>
{
	MaxpoolLayer<T>(int features, int rows, int cols, int maxed_rows, int maxed_cols)
	{
		feature_maps_count = feature_maps_num;
		out_rows = maxed_rows;
		out_cols = maxed_cols;
		out_features = feature_maps_count;
		in_rows = 0;
		in_cols = 0;
		in_feature = 0;
		Matrix2D<T> feature_sample(rows, cols);
		feature_maps = new[feature_maps_num * sizeof(feature_sample)];
		for (int i = 0; i < feature_maps_count)
			feature_maps[i] = feature_sample;
	}
	~MaxpoolLayer<T>()
	{
		delete[] feature_maps;
		delete[] data;
	}
	virtual Matrix2D<T>* feed_forwards()
	{
		Matrix2D<T>[feature_maps_count] output;
		for (int f = 0; f < feature_maps_count; ++f)
		{
			Matrix2D<Matrix2D<T>> samples;
			int down = feature_maps[f].rows / out_rows;
			int across = feature_maps[f].cos / out_cols;

			//get samples
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < out_cols; ++j)
					samples.at(i, j) = feature_maps[f].from(i * down, j * across, (i + 1) * down, (j + 1) * across);

			//find maxes
			Matrix2D<T> maxes(out_rows, out_cols);
			for (int i = 0; i < out_rows; ++i)
				for (int j = 0; j < out_cols; ++j)
					for (int n = 0; n < samples.at(i, j).rows; ++n)
						for (int m = 0; m < samples.at(i, j).cols; ++m)
							if (samples.at(i, j).at(n, m) > maxes.at(i, j))
								maxes.at(i, j) = samples.at(i, j).at(i, j)
			output[f] = maxes;
		}
		return output;
	}
};