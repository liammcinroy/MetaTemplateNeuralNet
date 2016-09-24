#pragma once

#include <vector>

#include "imatrix.h"

#define CNN_LAYER_INPUT 0
#define CNN_LAYER_CONVOLUTION 1
#define CNN_LAYER_PERCEPTRONFULLCONNECTIVITY 2
#define CNN_LAYER_BATCHNORMALIZATION 3
#define CNN_LAYER_MAXPOOL 4
#define CNN_LAYER_SOFTMAX 5
#define CNN_LAYER_OUTPUT 6

#define CNN_FUNC_LINEAR 0
#define CNN_FUNC_LOGISTIC 1
#define CNN_FUNC_BIPOLARLOGISTIC 2
#define CNN_FUNC_TANH 3
#define CNN_FUNC_TANHLECUN 4
#define CNN_FUNC_RELU 5
#define CNN_FUNC_RBM 6

#define CNN_BIAS_NONE 0
#define CNN_BIAS_CONV 1
#define CNN_BIAS_PERCEPTRON 2

#define CNN_DATA_FEATURE_MAP 0
#define CNN_DATA_WEIGHT_GRAD 1
#define CNN_DATA_BIAS_GRAD 2
#define CNN_DATA_WEIGHT_MOMENT 3
#define CNN_DATA_BIAS_MOMENT 4
#define CNN_DATA_WEIGHT_AUXDATA 5
#define CNN_DATA_BIAS_AUXDATA 6

// HELPER FUNCTIONS //// CLASS DEFINITIONS START AT LINE 395

template<size_t f, size_t r, size_t c, typename T = float> using FeatureMapVector = std::vector<FeatureMap<f, r, c, T>>;

template <size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s, bool use_pad> struct conv_helper_funcs
{
	static Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel);
	static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient);
	static Matrix2D<float, r, c> convolve_back(Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)>& input, Matrix2D<float, kernel_r, kernel_c>& kernel);
};

template<size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s> struct conv_helper_funcs<r, c, kernel_r, kernel_c, s, false>
{
	static Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		constexpr size_t out_r = (r - kernel_r) / s + 1;
		constexpr size_t out_c = (c - kernel_c) / s + 1;
		Matrix2D<float, out_r, out_c> output = { 0 };

		for (size_t i = N; i < (r - N); i += s)//change focus of kernel
		{
			for (size_t j = M; j < (c - M); j += s)
			{
				//iterate over kernel
				float sum = 0;
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						sum += input.at(i - n, j - m) * kernel.at(N - n, N - m);
				output.at((i - N) / s, (j - N) / s) = sum;
			}
		}
		return output;
	}

	static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		constexpr size_t out_r = (r - kernel_r) / s + 1;
		constexpr size_t out_c = (c - kernel_c) / s + 1;

		size_t i_0 = 0;
		size_t j_0 = 0;

		//change focus of kernel
		for (size_t i = N; i < (r - N); i += s)
		{
			for (size_t j = M; j < (c - M); j += s)
			{
				//iterate over kernel
				float sum = 0;
				float out = output.at(i_0, j_0);
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						kernel_gradient.at(N - n, M - m) += input.at(i - n, j - m) * out;
				++j_0;
			}
			j_0 = 0;
			++i_0;
		}
	}

	static Matrix2D<float, r, c> convolve_back(Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		Matrix2D<float, r, c> output = { 0 };

		size_t times_across = 0;
		size_t times_down = 0;

		for (size_t i = N; i < (r - N); i += s)
		{
			for (size_t j = M; j < (c - M); j += s)
			{
				//find all possible ways convolved size_to
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						output.at(i - n, j - m) += kernel.at(N - n, M - m) * input.at(times_down, times_across);
				++times_across;
			}
			times_across = 0;
			++times_down;
		}
		return output;
	}
};

template<size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s> struct conv_helper_funcs<r, c, kernel_r, kernel_c, s, true>
{
	static Matrix2D<float, r, c> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		constexpr size_t out_r = r;
		constexpr size_t out_c = c;
		Matrix2D<float, out_r, out_c> output = { 0 };

		//change focus of kernel
		for (size_t i = 0; i < r; i += s)
		{
			for (size_t j = 0; j < c; j += s)
			{
				//iterate over kernel
				float sum = 0;
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						sum += kernel.at(N - n, N - m) * (i < 0 || i >= r || j < 0 || j >= c ? 0 : input.at(i - n, j - m));
				output.at((i - N) / s, (j - N) / s) = sum;
			}
		}
		return output;
	}

	static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, r, c>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		constexpr size_t out_r = r;
		constexpr size_t out_c = c;

		size_t i_0 = 0;
		size_t j_0 = 0;

		//change focus of kernel
		for (size_t i = 0; i < r; i += s)
		{
			for (size_t j = 0; j < c; j += s)
			{
				//iterate over kernel
				float sum = 0;
				float out = output.at(i_0, j_0);
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						kernel_gradient.at(N - n, M - m) += out * (i < 0 || i >= r || j < 0 || j >= c ? 0 : input.at(i - n, j - m));
				++j_0;
			}
			j_0 = 0;
			++i_0;
		}
	}

	static Matrix2D<float, r, c> convolve_back(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
	{
		int N = (kernel_r - 1) / 2;
		int M = (kernel_c - 1) / 2;
		Matrix2D<float, r, c> output = { 0 };

		size_t times_across = 0;
		size_t times_down = 0;

		for (size_t i = 0; i < r; i += s)
		{
			for (size_t j = 0; j < c; j += s)
			{
				//find all possible ways convolved size_to
				for (int n = N; n >= -N; --n)
					for (int m = M; m >= -M; --m)
						output.at(i - n, j - m) += kernel.at(N - n, M - m) * (i < 0 || i >= r || j < 0 || j >= c ? 0 : input.at(times_down, times_across));
				++times_across;
			}
			times_across = 0;
			++times_down;
		}
		return output;
	}
};

//helper functions class
template<size_t feature, size_t row, size_t col> class Layer_Functions
{
public:
	using feature_maps_vector_type = FeatureMapVector<feature, row, col>;

	static void chain_activations(FeatureMap<feature, row, col>& fm, FeatureMap<feature, row, col>& o_fm, size_t activation)
	{
		for (size_t f = 0; f < feature; ++f)
			for (int i = 0; i < row; ++i)
				for (int j = 0; j < col; ++j)
					fm[f].at(i, j) *= activation_derivative(o_fm[f].at(i, j), activation);
	}

	static inline float activate(float value, size_t activation)
	{
		if (activation == CNN_FUNC_LINEAR)
			return value;
		else if (activation == CNN_FUNC_LOGISTIC || activation == CNN_FUNC_RBM)
			return value < 5 && value > -5 ? (1 / (1 + exp(-value))) : (value >= 5 ? 1.0f : 0.0f);
		else if (activation == CNN_FUNC_BIPOLARLOGISTIC)
			return value < 5 && value > -5 ? ((2 / (1 + exp(-value))) - 1) : (value >= 5 ? 1.0f : -1.0f);
		else if (activation == CNN_FUNC_TANH)
			return value < 5 && value > -5 ? tanh(value) : (value >= 5 ? 1.0f : -1.0f);
		else if (activation == CNN_FUNC_TANHLECUN)
			return value < 5 && value > -5 ? 1.7159f * tanh(0.66666667f * value) : ((value >= 5 ? 1.7159f : -1.7159f));
		else if (activation == CNN_FUNC_RELU)
			return value > 0 ? value : 0;
	}

	static inline float activation_derivative(float value, size_t activation)
	{
		if (activation == CNN_FUNC_LINEAR)
			return 1;
		else if (activation == CNN_FUNC_LOGISTIC || activation == CNN_FUNC_RBM)
			return value * (1 - value);
		else if (activation == CNN_FUNC_BIPOLARLOGISTIC)
			return (1 + value) * (1 - value) / 2;
		else if (activation == CNN_FUNC_TANH)
			return 1 - value * value;
		else if (activation == CNN_FUNC_TANHLECUN)
			return (0.66666667f / 1.7159f * (1.7159f + value) * (1.7159f - value));
		else if (activation == CNN_FUNC_RELU)
			return value > 0 ? 1.0f : 0.0f;
	}

	template<size_t f = feat, size_t r = row, size_t c = col>
	static inline void stochastic_sample(FeatureMap<f, r, c>& data)
	{
		if (activation == CNN_FUNC_RBM)
			for (size_t f = 0; f < f; ++f)
				for (size_t i = 0; i < r; ++i)
					for (size_t j = 0; j < c; ++j)
						data[f].at(i, j) = ((rand() * 1.0f) / RAND_MAX < data[f].at(i, j)) ? 1 : 0;
	}
};

//START ACTUAL LAYERS

template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> class ConvolutionLayer : public Layer_Functions<features, rows, cols>
{
public:
	ConvolutionLayer() = default;

	~ConvolutionLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<out_features, (use_padding ? rows : (rows - kernel_size) / stride + 1), (use_padding ? cols : (cols - kernel_size) / stride + 1)>& output)
	{
		constexpr size_t out_rows = use_padding ? rows : (rows - kernel_size) / stride + 1;
		constexpr size_t out_cols = use_padding ? cols : (cols - kernel_size) / stride + 1;

		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			//sum the kernels
			for (size_t f = 0; f < features; ++f)
			{
				add<float, out_rows, out_cols>(output[f_0],
					conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve(input[f], weights[f_0 * features + f]));
				if (use_biases)
					for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
						for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
							output[f_0].at(i_0, j_0) += biases[f_0 * features + f].at(0, 0);
			}

			if (activation_function != CNN_FUNC_LINEAR)
			{
				for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
					for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
						output[f_0].at(i_0, j_0) = activate(output[f_0].at(i_0, j_0), activation);
			}
		}
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<out_features, use_padding ? rows : (rows - kernel_size) / stride + 1, use_padding ? cols : (cols - kernel_size) / stride + 1>& input)
	{
		for (size_t f = 0; f < features; ++f)
		{
			for (size_t f_0 = 0; f_0 < out_features; ++f_0)
			{
				add<float, rows, cols>(output[f],
					conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve_back(input[f_0], weights[f_0 * features + f]));
			}

			for (size_t i = 0; i < rows; ++i)
			{
				for (size_t j = 0; j < cols; ++j)
				{
					if (use_biases && activation_function == CNN_FUNC_RBM)
						output[f].at(i, j) += generative_biases[f].at(i, j);
					output[f].at(i, j) = activate(input[f].at(i, j), activation_function);
				}
			}
		}
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<out_features, (use_padding ? rows : (rows - kernel_size) / stride + 1), (use_padding ? cols : (cols - kernel_size) / stride + 1)>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		constexpr size_t out_rows = use_padding ? rows : (rows - kernel_size) / stride + 1;
		constexpr size_t out_cols = use_padding ? cols : (cols - kernel_size) / stride + 1;

		//adjust gradients and update features
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t f = 0; f < features; ++f)
			{
				//update deltas
				add<float, rows, cols>(out_deriv[f],
					conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve_back(deriv[f_0], weights[f_0 * features + f]));

				//adjust the gradient
				conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::back_prop_kernel(activations_pre[f], deriv[f_0], weights_gradient[f_0 * features + f]);

				//L2 weight decay
				if (use_l2_weight_decay && online)
					for (size_t i = 0; i < kernel_size; ++i)
						for (size_t j = 0; j < kernel_size; ++j)
							weights_gradient[f_0 * features + f].at(i, j) += weights[f_0 * features + f].at(i, j);

				if (use_biases)
				{
					//normal derivative
					for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
						for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
							biases_gradient[f_0 * features + f].at(0, 0) += deriv[f_0].at(i_0, j_0);

					//l2 weight decay
					if (use_l2_weight_decay && include_biases_decay && online)
						biases_gradient[f_0 * features + f].at(0, 0) += 2 * weight_decay_factor * biases[f_0 * features + f].at(0, 0);
				}

				//update for online
				if (use_momentum && online)
				{
					for (size_t i = 0; i < kernel_size; ++i)
					{
						for (size_t j = 0; j < kernel_size; ++j)
						{
							weights[f_0 * features + f].at(i, j) += -learning_rate * (weights_gradient[f_0 * features + f].at(i, j) + momentum_term * weights_momentum[f_0 * features + f].at(i, j));
							weights_momentum[f_0 * features + f].at(i, j) = momentum_term * weights_momentum[f_0 * features + f].at(i, j) + weights_gradient[f_0 * features + f].at(i, j);
							weights_gradient[f_0 * features + f].at(i, j) = 0;
						}
					}

					if (use_biases)
					{
						biases[f_0 * features + f].at(0, 0) += -learning_rate * (biases_gradient[f_0 * features + f].at(0, 0) + momentum_term * biases_momentum[f_0 * features + f].at(0, 0));
						biases_momentum[f_0 * features + f].at(0, 0) = momentum_term * biases_momentum[f_0 * features + f].at(0, 0) + biases_gradient[f_0 * features + f].at(0, 0);
						biases_gradient[f_0 * features + f].at(0, 0) = 0;
					}
				}

				else if (online)
				{
					for (size_t i = 0; i < kernel_size; ++i)
					{
						for (size_t j = 0; j < kernel_size; ++j)
						{
							weights[f_0 * features + f].at(i, j) += -learning_rate * weights_gradient[f_0 * features + f].at(i, j);
							weights_gradient[f_0 * features + f].at(i, j) = 0;
						}
					}

					if (use_biases)
					{
						biases[f_0 * features + f].at(0, 0) += -learning_rate * biases_gradient[f_0 * features + f].at(0, 0);
						biases_gradient[f_0 * features + f].at(0, 0) = 0;
					}
				}
			}
		}

		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<out_features, (use_padding ? rows : (rows - kernel_size) / stride + 1), (use_padding ? cols : (cols - kernel_size) / stride + 1)>& outputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<out_features, use_padding ? rows : (rows - kernel_size) / stride + 1, use_padding ? cols : (cols - kernel_size) / stride + 1>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<out_features, (use_padding ? rows : (rows - kernel_size) / stride + 1), (use_padding ? cols : (cols - kernel_size) / stride + 1)>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static void wake_sleep(float& learning_rate, size_t markov_iterations, bool use_dropout)
	{
		constexpr size_t out_rows = use_padding ? rows : (rows - kernel_size) / stride + 1;
		constexpr size_t out_cols = use_padding ? cols : (cols - kernel_size) / stride + 1;

		//find difference via gibbs sampling
		FeatureMap<features, rows, cols> original = { 0 };
		for (size_t f = 0; f < features; ++f)
			original[f] = feature_maps[f].clone();

		FeatureMap<out_features, out_rows, out_cols> discriminated = { 0 };
		FeatureMap<out_features, out_rows, out_cols> reconstructed = { 0 };

		//Sample, but don't "normalize" second time
		feed_forwards(discriminated);
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
			reconstructed[f_0] = discriminated[f_0].clone();
		stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
		feed_backwards(reconstructed);
		if (!mean_field)
			stochastic_sample(feature_maps);
		feed_forwards(reconstructed);
		for (size_t its = 1; its < markov_iterations; ++its)
		{
			stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
			feed_backwards(reconstructed);
			if (!mean_field)
				stochastic_sample(feature_maps);
			feed_forwards(reconstructed);
		}

		constexpr size_t N = (kernel_size - 1) / 2;

		if (!mean_field)
			stochastic_sample<out_features, out_rows, out_cols>(discriminated);

		//adjust weights
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t f = 0; f < features; ++f)
			{
				size_t i = 0;
				size_t j = 0;

				if (!use_padding)
				{
					i = N;
					j = N;
				}

				for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
				{
					for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
					{
						for (int n = N; n >= -N; --n)
						{
							for (int m = N; m >= -N; --m)
							{
								float delta_weight = reconstructed[f_0].at(i_0, j_0) * feature_maps[f].at(i, j) - discriminated[f_0].at(i_0, j_0) * original[f].at(i, j);
								weights[f_0 * features + f].at(N - n, N - m) += -learning_rate * delta_weight;
							}
						}
						j += stride;
					}
					j = use_padding ? 0 : N;
					i += stride;
				}
			}

			//adjust hidden biases
			if (use_biases)
				for (size_t i_0 = 0; i_0 < biases[f_0].rows(); ++i_0)
					for (size_t j_0 = 0; j_0 < biases[f_0].cols(); ++j_0)
						biases[f_0].at(i_0, j_0) += -learning_rate * (reconstructed[f_0].at(i_0, j_0) - discriminated[f_0].at(i_0, j_0));
		}

		//adjust visible biases
		if (use_biases && activation_function == CNN_FUNC_RBM)
			for (size_t f = 0; f < features; ++f)
				for (size_t i = 0; i < rows; ++i)
					for (size_t j = 0; j < cols; ++j)
						generative_biases[f].at(i, j) += -learning_rate * (feature_maps[f].at(i, j) - original[f].at(i, j));
	}

	static constexpr size_t type = CNN_LAYER_CONVOLUTION;
	static constexpr size_t activation = activation_function;

	static size_t n;
	static bool mean_field;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases;
	static FeatureMap<out_features * features, kernel_size, kernel_size> weights;

	static FeatureMap<((use_biases && activation_function == CNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? cols : 0)> generative_biases;
	static FeatureMap<out_features * features, kernel_size, kernel_size> weights_aux_data;
	static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_aux_data;

	static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_gradient;
	static FeatureMap<out_features * features, kernel_size, kernel_size> weights_gradient;

	static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_momentum;
	static FeatureMap<out_features * features, kernel_size, kernel_size> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;
};

//initialize static
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> bool ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::mean_field = false;
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<features, rows, cols> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights = { -.1f, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<((use_biases && activation_function == CNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? cols : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::generative_biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<0, 0, 0> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<0, 0, 0> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> size_t ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> class PerceptronFullConnectivityLayer : public Layer_Functions<features, rows, cols>
{
public:
	PerceptronFullConnectivityLayer() = default;

	~PerceptronFullConnectivityLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<out_features, out_rows, out_cols>& output)
	{
		//loop through every neuron in output
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					//loop through every neuron in input and add it to output
					float sum = 0.0f;
					for (size_t f = 0; f < features; ++f)
						for (size_t i = 0; i < rows; ++i)
							for (size_t j = 0; j < cols; ++j)
								sum += (input[f].at(i, j) *
									weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j));

					//add bias
					if (use_biases)
						output[f_0].at(i_0, j_0) = activate(sum + biases[f_0].at(i_0, j_0), activation_function);
					else
						output[f_0].at(i_0, j_0) = activate(sum, activation_function);
				}
			}
		}
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<out_features, out_rows, out_cols>& input)
	{
		//go through every neuron in this layer
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t f = 0; f < features; ++f)
			{
				for (size_t i = 0; i < rows; ++i)
				{
					for (size_t j = 0; j < cols; ++j)
					{
						//go through every neuron in output layer and add it to this neuron
						float sum = 0.0f;
						for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
							for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
								sum += weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) * input[f_0].at(i_0, j_0);

						if (use_biases && activation_function == CNN_FUNC_RBM)
							sum += generative_biases[f].at(i, j);
						output[f].at(i, j) = activate(sum, activation_function);
					}
				}
			}
		}
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<out_features, out_rows, out_cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					if (use_biases)
					{
						//normal derivative
						biases_gradient[f_0].at(i_0, j_0) += deriv[f_0].at(i_0, j_0);

						//L2 weight decay
						if (use_l2_weight_decay && include_biases_decay && online)
							biases_gradient[f_0].at(i_0, j_0) += 2 * weight_decay_factor * biases[f_0].at(i_0, j_0);

						//online update
						if (use_momentum && online)
						{
							biases[f_0].at(i_0, j_0) += -learning_rate * (biases_gradient[f_0].at(i_0, j_0) + momentum_term * biases_momentum[f_0].at(i_0, j_0));
							biases_momentum[f_0].at(i_0, j_0) = momentum_term * biases_momentum[f_0].at(i_0, j_0) + biases_gradient[f_0].at(i_0, j_0);
							biases_gradient[f_0].at(i_0, j_0) = 0;
						}

						else if (online)
						{
							biases[f_0].at(i_0, j_0) += -learning_rate * biases_gradient[f_0].at(i_0, j_0);
							biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}

					for (size_t f = 0; f < features; ++f)
					{
						for (size_t i = 0; i < rows; ++i)
						{
							for (size_t j = 0; j < cols; ++j)
							{
								//update deltas
								out_deriv[f].at(i, j) += deriv[f_0].at(i_0, j_0) * weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);

								//normal derivative
								weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += deriv[f_0].at(i_0, j_0) * activations_pre[f].at(i, j);

								//L2 decay
								if (use_l2_weight_decay && online)
									weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += 2 * weight_decay_factor * weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);

								//Online updates
								if (use_momentum && online)
								{
									weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += -learning_rate * (weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) + momentum_term * weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j));
									weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = momentum_term * weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) + weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);
									weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = 0;
								}

								else if (online)
								{
									weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) +=
										-learning_rate * weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);
									weights_gradient[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = 0;
								}
							}
						}
					}
				}
			}
		}

		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<out_features, out_rows, out_cols>& outputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<out_features, out_rows, out_cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<out_features, out_rows, out_cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static void wake_sleep(float& learning_rate, size_t markov_iterations, bool use_dropout)
	{
		//find difference via gibbs sampling
		FeatureMap<features, rows, cols> original = { 0 };

		FeatureMap<out_features, out_rows, out_cols> discriminated = { 0 };

		FeatureMap<out_features, out_rows, out_cols> reconstructed = { 0 };

		//Sample, but don't "normalize" second time
		feed_forwards(discriminated);
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
			reconstructed[f_0] = discriminated[f_0].clone();
		stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
		feed_backwards(reconstructed);
		if (!mean_field)
			stochastic_sample(feature_maps);
		feed_forwards(reconstructed);
		for (size_t its = 1; its < markov_iterations; ++its)
		{
			stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
			feed_backwards(reconstructed);
			if (!mean_field)
				stochastic_sample(feature_maps);
			feed_forwards(reconstructed);
		}

		if (!mean_field)
			stochastic_sample<out_features, out_rows, out_cols>(discriminated);

		//adjust weights
		for (size_t f_0 = 0; f_0 < out_features; ++f_0)
		{
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					for (size_t f = 0; f < features; ++f)
					{
						for (size_t i = 0; i < rows; ++i)
						{
							for (size_t j = 0; j < cols; ++j)
							{
								float delta_weight = reconstructed[f_0].at(i_0, j_0) * feature_maps[f].at(i, j) - discriminated[f_0].at(i_0, j_0) * original[f].at(i, j);
								weights[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += -learning_rate * delta_weight;
							}
						}
					}
				}
			}

			//adjust hidden biases
			if (use_biases)
				for (size_t i_0 = 0; i_0 < biases[f_0].rows(); ++i_0)
					for (size_t j_0 = 0; j_0 < biases[f_0].cols(); ++j_0)
						biases[f_0].at(i_0, j_0) += -learning_rate * (reconstructed[f_0].at(i_0, j_0) - discriminated[f_0].at(i_0, j_0));
		}

		//adjust visible biases
		if (use_biases && activation_function == CNN_FUNC_RBM)
			for (size_t f = 0; f < features; ++f)
				for (size_t i = 0; i < rows; ++i)
					for (size_t j = 0; j < cols; ++j)
						generative_biases[f].at(i, j) += -learning_rate * (feature_maps[f].at(i, j) - original[f].at(i, j));
	}

	static constexpr size_t type = CNN_LAYER_PERCEPTRONFULLCONNECTIVITY;
	static constexpr size_t activation = activation_function;

	static size_t n;
	static bool mean_field;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases;
	static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights;

	static FeatureMap<((use_biases && activation_function == CNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? cols : 0)> generative_biases;
	static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_aux_data;
	static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_aux_data;

	static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_gradient;
	static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_gradient;

	static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_momentum;
	static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;
};

//static variable initialization
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> bool PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::mean_field = false;
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<features, rows, cols> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights = { -.1f, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<((use_biases && activation_function == CNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == CNN_FUNC_RBM) ? cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::generative_biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<0, 0, 0> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<0, 0, 0> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> size_t PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> class BatchNormalizationLayer : public Layer_Functions<features, rows, cols>
{
public:
	BatchNormalizationLayer() = default;

	~BatchNormalizationLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<features, rows, cols>& output)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = activate(weights[f].at(i, j) * (input[f].at(i, j) - activations_population_mean[f].at(i, j)) / sqrt(activations_population_variance[f].at(i, j) + min_divisor) + biases[f].at(i, j), activation_function);
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<features, rows, cols>& input)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)//todo?
					output[f].at(i, j) = weights[f].at(i, j) * (input[f].at(i, j) - activations_population_mean[f].at(i, j)) / sqrt(activations_population_variance[f].at(i, j) + min_divisor) + biases[f].at(i, j);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<features, rows, cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		//undefined for single 
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<features, rows, cols>& outputs, bool discriminating = false)
	{
		//different output for training
		for (size_t f = 0; f < features; ++f)
		{
			for (size_t i = 0; i < rows; ++i)
			{
				for (size_t j = 0; j < cols; ++j)
				{
					float sumx = 0.0f;
					float sumxsqr = 0.0f;
					size_t n_in = outputs.size();
					//compute statistics
					for (size_t in = 0; in < n_in; ++in)
					{
						float x = inputs[in][f].at(i, j);
						sumx += x;
						sumxsqr += x * x;
					}

					//apply to outputs
					float mean = sumx / n_in;
					float var = sumxsqr / n_in - mean * mean;
					float std = sqrt(var + min_divisor);
					float gamma = weights[f].at(i, j);
					float beta = biases[f].at(i, j);
					for (size_t in = 0; in < n_in; ++in)
						outputs[in][f].at(i, j) = activate(gamma * (inputs[in][f].at(i, j) - mean) / std + beta, activation_function);

					/*activations_population_mean[f].at(i, j) = mean;
					activations_population_variance[f].at(i, j) = var;*/ //keeps relatively stable batch vs discriminatory values

					//set minibatch stats
					biases_aux_data[f].at(i, j) = mean;
					weights_aux_data[f].at(i, j) = var;

					//update population statistics
					if (n == 0)
					{
						activations_population_mean[f].at(i, j) = mean;
						activations_population_variance[f].at(i, j) = var;
					}

					else
					{
						float old_mean = activations_population_mean[f].at(i, j);
						float old_var = activations_population_variance[f].at(i, j);
						float momentum = .8f;
						activations_population_mean[f].at(i, j) = (1 - momentum) * mean + momentum * old_mean;//(old_mean * n + mean) / (n + 1);
						activations_population_variance[f].at(i, j) = (1 - momentum) * var + momentum * old_var;//(old_var * (n - 1) / n + var) * n / (n + 1);
					}

					++n; //fails completely due to / n w/ n == 0
				}
			}
		}
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<features, rows, cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<features, rows, cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
		{
			auto& deriv = derivs[in];
			auto& out_deriv = out_derivs[in];
			auto& activations_pre = activations_pre_vec[in];
			for (size_t f = 0; f < features; ++f)
			{
				for (size_t i = 0; i < rows; ++i)
				{
					for (size_t j = 0; j < cols; ++j)
					{
						float mu = biases_aux_data[f].at(i, j); //todo: change from population to minibatch
						float div = activations_pre[f].at(i, j) - mu;
						float std = sqrt(weights_aux_data[f].at(i, j) + min_divisor);

						float xhat = div / std;
						float d_out = deriv[f].at(i, j);

						biases_gradient[f].at(i, j) += d_out;
						weights_gradient[f].at(i, j) += d_out * xhat;

						float sumDeriv = 0.0f;
						float sumDiff = 0.0f;
						for (size_t in2 = 0; in2 < out_derivs.size(); ++in2)
						{
							float d_outj = out_derivs[in2][f].at(i, j);
							sumDeriv += d_outj;
							sumDiff += d_outj * (activations_pre_vec[in2][f].at(i, j) - mu);
						}

						out_deriv[f].at(i, j) = weights[f].at(i, j) / derivs.size() / std * (derivs.size() * d_out - sumDeriv - div / std * sumDiff);
					}
				}
			}

			//apply derivatives
			chain_activations(out_deriv, activations_pre, previous_layer_activation);
		}
	}

	static const float min_divisor;
	static size_t n;

	static constexpr size_t type = CNN_LAYER_BATCHNORMALIZATION;
	static constexpr size_t activation = activation_function;

	static FeatureMap<features, rows, cols> feature_maps;

	static FeatureMap<features, rows, cols> activations_population_mean;
	static FeatureMap<features, rows, cols> activations_population_variance;

	static FeatureMap<features, rows, cols> biases; //beta
	static FeatureMap<features, rows, cols> weights; //gamma

	static FeatureMap<features, rows, cols> biases_gradient;
	static FeatureMap<features, rows, cols> weights_gradient;

	static FeatureMap<0, 0, 0> generative_biases;
	static FeatureMap<features, rows, cols> biases_aux_data;
	static FeatureMap<features, rows, cols> weights_aux_data;

	static FeatureMap<features, rows, cols> biases_momentum;
	static FeatureMap<features, rows, cols> weights_momentum;
};

//init static
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::weights = { 1 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<features, rows, cols> BatchNormalizationLayer<index, features, rows, cols, activation_function>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> FeatureMap<0, 0, 0> BatchNormalizationLayer<index, features, rows, cols, activation_function>::generative_biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> const float BatchNormalizationLayer<index, features, rows, cols, activation_function>::min_divisor = .0001f;
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> size_t BatchNormalizationLayer<index, features, rows, cols, activation_function>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> class MaxpoolLayer : public Layer_Functions<features, rows, cols>
{
public:
	MaxpoolLayer() = default;

	~MaxpoolLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<features, out_rows, out_cols>& output)
	{
		for (size_t f_0 = 0; f_0 < features; ++f_0)
			for (size_t i = 0; i < output[f_0].rows(); ++i)
				for (size_t j = 0; j < output[f_0].cols(); ++j)
					output[f_0].at(i, j) = -INFINITY;

		for (size_t f = 0; f < features; ++f)
		{
			constexpr size_t down = rows / out_rows;
			constexpr size_t across = cols / out_cols;
			Matrix2D<Matrix2D<float, down, across>, out_rows, out_cols> samples;

			//get samples
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					//get the current sample
					size_t maxI = (i_0 + 1) * down;
					size_t maxJ = (j_0 + 1) * across;
					for (size_t i2 = i_0 * down; i2 < maxI; ++i2)
					{
						for (size_t j2 = j_0 * across; j2 < maxJ; ++j2)
						{
							samples.at(i_0, j_0).at(maxI - i2 - 1, maxJ - j2 - 1) = input[f].at(i2, j2);
						}
					}
				}
			}

			//find maxes
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					for (size_t n = 0; n < samples.at(i_0, j_0).rows(); ++n)
					{
						for (size_t m = 0; m < samples.at(i_0, j_0).cols(); ++m)
						{
							if (samples.at(i_0, j_0).at(n, m) > output[f].at(i_0, j_0))
							{
								output[f].at(i_0, j_0) = samples.at(i_0, j_0).at(n, m);
								switches[f].at(i_0, j_0) = std::make_pair(n, m);
							}
						}
					}
				}
			}
		}
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<features, out_rows, out_cols>& input)
	{
		for (size_t f = 0; f < features; ++f)
		{
			size_t down = rows / out_rows;
			size_t across = cols / out_cols;

			//search each sample
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					std::pair<size_t, size_t> coords = switches[f].at(i_0, j_0);
					for (size_t i = 0; i < down; ++i)
					{
						for (size_t j = 0; j < across; ++j)
						{
							if (i == coords.first && j == coords.second)
								output[f].at(i_0 * down + i, j_0 * across + j) = input[f].at(i_0, j_0);
							else
								output[f].at(i * down, j * across) = 0;
						}
					}
				}
			}
		}
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<features, out_rows, out_cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		//just move the values back to which ones were passed on
		for (size_t f = 0; f < features; ++f)
		{
			size_t down = rows / out_rows;
			size_t across = cols / out_cols;

			//search each sample
			for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
			{
				for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
				{
					std::pair<size_t, size_t> coords = switches[f].at(i_0, j_0);
					for (size_t i = 0; i < down; ++i)
					{
						for (size_t j = 0; j < across; ++j)
						{
							if (i == coords.first && j == coords.second)
								out_deriv[f].at(i_0 * down + i, j_0 * across + j) = deriv[f].at(i_0, j_0);
							else
								out_deriv[f].at(i * down, j * across) = 0;
						}
					}
				}
			}
		}

		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<features, out_rows, out_cols>& outputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<features, out_rows, out_cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<features, out_rows, out_cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static constexpr size_t type = CNN_LAYER_MAXPOOL;
	static constexpr size_t activation = CNN_FUNC_LINEAR;

	static size_t n;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<0, 0, 0> biases;
	static FeatureMap<0, 0, 0> weights;

	static FeatureMap<0, 0, 0> generative_biases;
	static FeatureMap<0, 0, 0> weights_aux_data;
	static FeatureMap<0, 0, 0> biases_aux_data;

	static FeatureMap<0, 0, 0> biases_gradient;
	static FeatureMap<0, 0, 0> weights_gradient;

	static FeatureMap<0, 0, 0> biases_momentum;
	static FeatureMap<0, 0, 0> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;

private:
	static FeatureMap<features, out_rows, out_cols, std::pair<size_t, size_t>> switches;
};

//init static
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<features, rows, cols> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::weights = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::generative_biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<0, 0, 0> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> FeatureMap<features, out_rows, out_cols, std::pair<size_t, size_t>> MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::switches = {};
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> size_t MaxpoolLayer<index, features, rows, cols, out_rows, out_cols>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols> class SoftMaxLayer : public Layer_Functions<features, rows, cols>
{
public:
	SoftMaxLayer() = default;

	~SoftMaxLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<features, rows, cols>& output)
	{
		for (size_t f = 0; f < features; ++f)
		{
			//find total
			float sum = 0.0f;
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					sum += input[f].at(i, j) < 6 ? exp(input[f].at(i, j)) : 4;

			//get prob
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = exp(input[f].at(i, j)) / sum;
		}
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<features, rows, cols>& input)
	{
		//assume that the original input has a mean of 0, so sum of original input would *approximately* be the total number of inputs
		size_t total = rows * cols;

		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = log(total * input[f].at(i, j));
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<features, rows, cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t f = 0; f < features; ++f)
		{
			for (size_t i = 0; i < rows; ++i)
			{
				for (size_t j = 0; j < cols; ++j)
				{
					//cycle through all again
					for (size_t i2 = 0; i2 < rows; ++i2)
					{
						for (size_t j2 = 0; j2 < cols; ++j2)
						{
							/*float h_i = data[f].at(i, j);
							float h_j = data[f].at(i2, j2);
							feature_maps[f].at(i, j) += (i2 == i && j2 == j) ? h_i * (1 - h_i) : -h_i * h_j;*///todo: check
						}
					}
				}
			}
		}
		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<features, rows, cols>& outputs, bool discriminating = false)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<features, rows, cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<features, rows, cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static constexpr size_t type = CNN_LAYER_SOFTMAX;
	static constexpr size_t activation = CNN_FUNC_LINEAR;

	static size_t n;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<0, 0, 0> biases;
	static FeatureMap<0, 0, 0> weights;

	static FeatureMap<0, 0, 0> generative_biases;
	static FeatureMap<0, 0, 0> weights_aux_data;
	static FeatureMap<0, 0, 0> biases_aux_data;

	static FeatureMap<0, 0, 0> biases_gradient;
	static FeatureMap<0, 0, 0> weights_gradient;

	static FeatureMap<0, 0, 0> biases_momentum;
	static FeatureMap<0, 0, 0> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;
};

//init static
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<features, rows, cols> SoftMaxLayer<index, features, rows, cols>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::weights = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::generative_biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> SoftMaxLayer<index, features, rows, cols>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> size_t SoftMaxLayer<index, features, rows, cols>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols> class InputLayer : public Layer_Functions<features, rows, cols>
{
public:
	InputLayer() = default;

	~InputLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<features, rows, cols>& output)
	{
		//just output
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = input[f].at(i, j);
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<features, rows, cols>& input)
	{
		//just output
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = input[f].at(i, j);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<features, rows, cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					out_deriv[f].at(i, j) = deriv[f].at(i, j);
		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<features, rows, cols>& outputs, bool discriminating = false)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<features, rows, cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<features, rows, cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static constexpr size_t type = CNN_LAYER_INPUT;
	static constexpr size_t activation = CNN_FUNC_LINEAR;

	static size_t n;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<0, 0, 0> biases;
	static FeatureMap<0, 0, 0> weights;

	static FeatureMap<0, 0, 0> generative_biases;
	static FeatureMap<0, 0, 0> weights_aux_data;
	static FeatureMap<0, 0, 0> biases_aux_data;

	static FeatureMap<0, 0, 0> biases_gradient;
	static FeatureMap<0, 0, 0> weights_gradient;

	static FeatureMap<0, 0, 0> biases_momentum;
	static FeatureMap<0, 0, 0> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;
};

//init static
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<features, rows, cols> InputLayer<index, features, rows, cols>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::weights = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::generative_biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> InputLayer<index, features, rows, cols>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> size_t InputLayer<index, features, rows, cols>::n = 0;

template<size_t index, size_t features, size_t rows, size_t cols> class OutputLayer : public Layer_Functions<features, rows, cols>
{
public:
	OutputLayer() = default;

	~OutputLayer() = default;

	static void feed_forwards(FeatureMap<features, rows, cols>& input, FeatureMap<features, rows, cols>& output)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = input[f].at(i, j);
	}

	static void feed_backwards(FeatureMap<features, rows, cols>& output, FeatureMap<features, rows, cols>& input)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					output[f].at(i, j) = input[f].at(i, j);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMap<features, rows, cols>& deriv, FeatureMap<features, rows, cols>& activations_pre, FeatureMap<features, rows, cols>& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t f = 0; f < features; ++f)
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
					out_deriv[f].at(i, j) = deriv[f].at(i, j);
		//apply derivatives
		chain_activations(out_deriv, activations_pre, previous_layer_activation);
	}

	static void feed_forwards(FeatureMapVector<features, rows, cols>& inputs, FeatureMapVector<features, rows, cols>& outputs, bool discriminating = false)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_forwards(inputs[in], outputs[in]);
	}

	static void feed_backwards(FeatureMapVector<features, rows, cols>& outputs, FeatureMapVector<features, rows, cols>& inputs)
	{
		for (size_t in = 0; in < outputs.size(); ++in)
			feed_backwards(outputs[in], inputs[in]);
	}

	static void back_prop(size_t previous_layer_activation, FeatureMapVector<features, rows, cols>& derivs, FeatureMapVector<features, rows, cols>& activations_pre_vec, FeatureMapVector<features, rows, cols>& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor)
	{
		for (size_t in = 0; in < derivs.size(); ++in)
			back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor);
	}

	static constexpr size_t type = CNN_LAYER_OUTPUT;
	static constexpr size_t activation = CNN_FUNC_LINEAR;

	static size_t n;

	static FeatureMap<features, rows, cols> feature_maps;
	static FeatureMap<0, 0, 0> biases;
	static FeatureMap<0, 0, 0> weights;

	static FeatureMap<0, 0, 0> generative_biases;
	static FeatureMap<0, 0, 0> weights_aux_data;
	static FeatureMap<0, 0, 0> biases_aux_data;

	static FeatureMap<0, 0, 0> biases_gradient;
	static FeatureMap<0, 0, 0> weights_gradient;

	static FeatureMap<0, 0, 0> biases_momentum;
	static FeatureMap<0, 0, 0> weights_momentum;

	static FeatureMap<0, 0, 0> activations_population_mean;
	static FeatureMap<0, 0, 0> activations_population_variance;
};

//init static
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<features, rows, cols> OutputLayer<index, features, rows, cols>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::weights = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::generative_biases = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> FeatureMap<0, 0, 0> OutputLayer<index, features, rows, cols>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols> size_t OutputLayer<index, features, rows, cols>::n = 0;