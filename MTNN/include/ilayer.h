#pragma once

#include "imatrix.h"

////All of the types etc.

//Will be defined in a Layer class' "type" variable
#define MTNN_LAYER_INPUT 0
#define MTNN_LAYER_CONVOLUTION 1
#define MTNN_LAYER_PERCEPTRONFULLCONNECTIVITY 2
#define MTNN_LAYER_BATCHNORMALIZATION 3
#define MTNN_LAYER_MAXPOOL 4
#define MTNN_LAYER_SOFTMAX 5
#define MTNN_LAYER_OUTPUT 6

//WIP
#define MTNN_LAYER_LSTM 7

//Use as definition for "activation_function" parameter (if applicable)
#define MTNN_FUNC_LINEAR 0
#define MTNN_FUNC_LOGISTIC 1
#define MTNN_FUNC_BIPOLARLOGISTIC 2
#define MTNN_FUNC_TANH 3
#define MTNN_FUNC_TANHLECUN 4
#define MTNN_FUNC_RELU 5
#define MTNN_FUNC_RBM 6

//used in cleanup in neural net class
#define MTNN_DATA_FEATURE_MAP 0
#define MTNN_DATA_WEIGHT_GRAD 1
#define MTNN_DATA_BIAS_GRAD 2
#define MTNN_DATA_WEIGHT_MOMENT 3
#define MTNN_DATA_BIAS_MOMENT 4
#define MTNN_DATA_WEIGHT_AUXDATA 5
#define MTNN_DATA_BIAS_AUXDATA 6

//// HELPER FUNCTIONS //// CLASS DEFINITIONS START AT LINE 395

template<size_t f, size_t r, size_t c, typename T = float> using FeatureMapVector = std::vector<FeatureMap<f, r, c, T>>;

//abstract class for padding, non padding variants; even or odd kernels shouldn't matter
template <size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s, bool use_pad> struct conv_helper_funcs
{
    static Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel);
    static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient);
    static Matrix2D<float, r, c> convolve_back(Matrix2D<float, (use_pad ? r : (r - kernel_r) / s + 1), (use_pad ? c : (c - kernel_c) / s + 1)>& input, Matrix2D<float, kernel_r, kernel_c>& kernel);
};

//no padding specifications (most common)
template<size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s> struct conv_helper_funcs<r, c, kernel_r, kernel_c, s, false>
{
    //feed forward
    static Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
    {
        constexpr size_t out_r = (r - kernel_r) / s + 1;
        constexpr size_t out_c = (c - kernel_c) / s + 1;
        Matrix2D<float, out_r, out_c> output = { 0 };

        for (size_t i = 0; i < r - kernel_r; i += s)//change top left of overlayed kernel
        {
            for (size_t j = 0; j < c - kernel_c; j += s)
            {
                //iterate over kernel
                float sum = 0;
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        sum += input.at(i + n, j + m) * kernel.at(n, m);
                output.at(i / s, j / s) = sum;
            }
        }
        return output;
    }

    //accumulates gradients on kernel
    static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient)
    {
        constexpr size_t out_r = (r - kernel_r) / s + 1;
        constexpr size_t out_c = (c - kernel_c) / s + 1;

        size_t i_0 = 0;
        size_t j_0 = 0;

        //change focus of kernel
        for (size_t i = 0; i < r - kernel_r; i += s)//change top left of overlayed kernel
        {
            for (size_t j = 0; j < c - kernel_c; j += s)
            {
                //iterate over kernel
                float sum = 0;
                float out = output.at(i_0, j_0);
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        kernel_gradient.at(n, m) += input.at(i + n, j + m) * out;
                ++j_0;
            }
            j_0 = 0;
            ++i_0;
        }
    }

    //feed back
    static Matrix2D<float, r, c> convolve_back(Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
    {
        Matrix2D<float, r, c> output = { 0 };

        size_t i_0 = 0;
        size_t j_0 = 0;

        for (size_t i = 0; i < r - kernel_r; i += s)//change top left of overlayed kernel
        {
            for (size_t j = 0; j < c - kernel_c; j += s)
            {
                //find all possible ways convolved size_to
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        output.at(i + n, j + m) += kernel.at(n, m) * input.at(i_0, j_0);
                ++j_0;
            }
            j_0 = 0;
            ++i_0;
        }
        return output;
    }
};

//padding variants
template<size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s> struct conv_helper_funcs<r, c, kernel_r, kernel_c, s, true>
{
    //feed forward
    static Matrix2D<float, r, c> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
    {
        int N = (kernel_r - 1) / 2;
        int M = (kernel_c - 1) / 2;
        constexpr size_t out_r = r;
        constexpr size_t out_c = c;
        Matrix2D<float, out_r, out_c> output = { 0 };
        
        //change top left of kernel
        for (int i = -N; i < (int)r - N; i += s)
        {
            for (int j = -M; j < (int)c - M; j += s)
            {
                //iterate over kernel
                float sum = 0;
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        sum += kernel.at(n, m) * (i + n < 0 || i + n >= r || j + m < 0 || j + m >= c ? 0 : input.at(i + n, j + m));
                output.at((i + M) / s, (j + M) / s) = sum;
            }
        }
        return output;
    }

    //accumulate gradients of kernel
    static void back_prop_kernel(Matrix2D<float, r, c>& input, Matrix2D<float, r, c>& output, Matrix2D<float, kernel_r, kernel_c>& kernel_gradient)
    {
        int N = (kernel_r - 1) / 2;
        int M = (kernel_c - 1) / 2;
        constexpr size_t out_r = r;
        constexpr size_t out_c = c;

        size_t i_0 = 0;
        size_t j_0 = 0;

        //change top left of kernel
        for (int i = -N; i < (int)r - N; i += s)
        {
            for (int j = -M; j < (int)c - M; j += s)
            {
                //iterate over kernel
                float sum = 0;
                float out = output.at(i_0, j_0);
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        kernel_gradient.at(n, m) += out * (i + n < 0 || i + n >= r || j + m < 0 || j + m >= c ? 0 : input.at(i + n, j + m));
                ++j_0;
            }
            j_0 = 0;
            ++i_0;
        }
    }

    //feed back
    static Matrix2D<float, r, c> convolve_back(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
    {
        int N = (kernel_r - 1) / 2;
        int M = (kernel_c - 1) / 2;
        Matrix2D<float, r, c> output = { 0 };

        size_t i_0 = 0;
        size_t j_0 = 0;

        //change top left of kernel
        for (int i = -N; i < (int)r - N; i += s)
        {
            for (int j = -M; j < (int)c - M; j += s)
            {
                //find all possible ways convolved size_to
                for (int n = 0; n < kernel_r; n++)
                    for (int m = 0; m < kernel_c; m++)
                        output.at(i + n, j + m) += kernel.at(n, m) * (i + n < 0 || i + n >= r || j + m < 0 || j + m >= c ? 0 : input.at(i_0, j_0));
                ++j_0;
            }
            j_0 = 0;
            ++i_0;
        }
        return output;
    }
};

//helper functions class - defines actions used in all classes (like activations, chain rule, etc.)
template<size_t feature, size_t row, size_t col> class Layer_Functions
{
public:
    using feature_maps_type = FeatureMap<feature, row, col>;
    using feature_maps_vector_type = FeatureMapVector<feature, row, col>;

    //apply chain rule (store in fm, output of feed forward as o_fm)
    static void chain_activations(FeatureMap<feature, row, col>& fm, FeatureMap<feature, row, col>& o_fm, size_t activation)
    {
        for (size_t f = 0; f < feature; ++f)
            for (int i = 0; i < row; ++i)
                for (int j = 0; j < col; ++j)
                    fm[f].at(i, j) *= activation_derivative(o_fm[f].at(i, j), activation);
    }

    //returns the activation of an input
    static inline float activate(float value, size_t activation)
    {
        if (activation == MTNN_FUNC_LINEAR)
            return value;
        else if (activation == MTNN_FUNC_LOGISTIC || activation == MTNN_FUNC_RBM)
            return value < 5 && value > -5 ? (1 / (1 + exp(-value))) : (value >= 5 ? 1.0f : 0.0f);
        else if (activation == MTNN_FUNC_BIPOLARLOGISTIC)
            return value < 5 && value > -5 ? ((2 / (1 + exp(-value))) - 1) : (value >= 5 ? 1.0f :  - 1.0f);
        else if (activation == MTNN_FUNC_TANH)
            return value < 5 && value > -5 ? tanh(value) : (value >= 5 ? 1.0f :  - 1.0f);
        else if (activation == MTNN_FUNC_TANHLECUN)
            return value < 5 && value > -5 ? 1.7159f * tanh(0.66666667f * value) : ((value >= 5 ? 1.7159f :  - 1.7159f));
        else if (activation == MTNN_FUNC_RELU)
            return value > 0 ? value : 0;
    }

    //derivative of activation function (pass in the output of the activation function)
    static inline float activation_derivative(float value, size_t activation)
    {
        if (activation == MTNN_FUNC_LINEAR)
            return 1;
        else if (activation == MTNN_FUNC_LOGISTIC || activation == MTNN_FUNC_RBM)
            return value * (1 - value);
        else if (activation == MTNN_FUNC_BIPOLARLOGISTIC)
            return (1 + value) * (1 - value) / 2;
        else if (activation == MTNN_FUNC_TANH)
            return 1 - value * value;
        else if (activation == MTNN_FUNC_TANHLECUN)
            return (0.66666667f / 1.7159f * (1.7159f + value) * (1.7159f - value));
        else if (activation == MTNN_FUNC_RELU)
            return value > 0 ? 1.0f : 0.0f;
    }

    //use to sample an RBM (each cell is independent of others)
    template<size_t f, size_t r, size_t c>
    static inline void stochastic_sample(FeatureMap<f, r, c>& data)
    {
        for (size_t f = 0; f < f; ++f)
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j)
                    data[f].at(i, j) = ((rand() * 1.0f) / RAND_MAX < data[f].at(i, j)) ? 1 : 0;
    }
};

////START ACTUAL LAYERS

//Convolutional layer: use even or odd kernels (always square), padding or not (zero padding if true), biases or not (should almost always be true), any stride, any out_features, any activation function
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> class ConvolutionLayer : public Layer_Functions<features, rows, cols>
{
public:

    //not used except batch norm
    static size_t n;

    //used in wake-sleep only
    static bool mean_field;

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    //biases (if used) are kept in own matrix
    static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases;
    //kernels
    static FeatureMap<out_features * features, kernel_size, kernel_size> weights;
    //only used in wake-sleep/feed back/rbms
    static FeatureMap<((use_biases && activation_function == MTNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? cols : 0)> generative_biases;

    //used for hessian (old) or Adam
    static FeatureMap<out_features * features, kernel_size, kernel_size> weights_aux_data;
    //used for hessian (old) or Adam
    static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_aux_data;

    //stores actual gradient
    static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_gradient;
    //stores actual gradient
    static FeatureMap<out_features * features, kernel_size, kernel_size> weights_gradient;

    //stores momentum (if applicable)
    static FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> biases_momentum;
    //stores momentum (if applicable)
    static FeatureMap<out_features * features, kernel_size, kernel_size> weights_momentum;

    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_mean;
    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_CONVOLUTION;
    //activation function type (dynamic test, but not stored since constexpr)
    static constexpr size_t activation = activation_function;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<out_features, use_padding ? rows / stride + 1 : (rows - kernel_size) / stride + 1, use_padding ? cols / stride + 1 : (cols - kernel_size) / stride + 1>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //never used, static class
    ConvolutionLayer() = default;

    //never used, static class
    ~ConvolutionLayer() = default;

    //feed forwards given input, weights, biases to output
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        constexpr size_t out_rows = use_padding ? rows / stride + 1 : (rows - kernel_size) / stride + 1;
        constexpr size_t out_cols = use_padding ? cols / stride + 1 : (cols - kernel_size) / stride + 1;

        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            //sum the kernels
            for (size_t f = 0; f < features; ++f)
            {
                //sum all convolutions from previous feature maps and put in output map
                add<float, out_rows, out_cols>(output[f_0],
                    conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve(input[f], params_w[f_0 * features + f]));
                //add bias (if applicable)
                if (use_biases)
                    for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                        for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                            output[f_0].at(i_0, j_0) += params_b[f_0 * features + f].at(0, 0);
            }

            //apply activation (if not linear)
            if (activation_function != MTNN_FUNC_LINEAR)
                for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                    for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                        output[f_0].at(i_0, j_0) = activate(output[f_0].at(i_0, j_0), activation);
        }
    }

    //undo feed forwards, with generative biases instead
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t f = 0; f < features; ++f)
        {
            //go backwards over all output maps to current map
            for (size_t f_0 = 0; f_0 < out_features; ++f_0)
                add<float, rows, cols>(output[f],
                    conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve_back(input[f_0], params_w[f_0 * features + f]));

            //activate if necessary
            if (activation_function != MTNN_FUNC_LINEAR)
            {
                for (size_t i = 0; i < rows; ++i)
                {
                    for (size_t j = 0; j < cols; ++j)
                    {
                        if (use_biases && activation_function == MTNN_FUNC_RBM)
                            output[f].at(i, j) += params_b[f].at(i, j);
                        output[f].at(i, j) = activate(input[f].at(i, j), activation_function);
                    }
                }
            }
        }
    }

    //accumulate gradients in given, using given weights, biases, outputs, activations, derivs, etc. WILL APPLY IF ONLINE LEARNING and vanilla backprop
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        constexpr size_t out_rows = use_padding ? rows / stride + 1 : (rows - kernel_size) / stride + 1;
        constexpr size_t out_cols = use_padding ? cols / stride + 1 : (cols - kernel_size) / stride + 1;

        //adjust gradients and update features
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            for (size_t f = 0; f < features; ++f)
            {
                //update deltas
                add<float, rows, cols>(out_deriv[f],
                    conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::convolve_back(deriv[f_0], params_w[f_0 * features + f]));

                //adjust the gradient
                conv_helper_funcs<rows, cols, kernel_size, kernel_size, stride, use_padding>::back_prop_kernel(activations_pre[f], deriv[f_0], w_grad[f_0 * features + f]);

                //L2 weight decay
                if (use_l2_weight_decay && online)
                    for (size_t i = 0; i < kernel_size; ++i)
                        for (size_t j = 0; j < kernel_size; ++j)
                            w_grad[f_0 * features + f].at(i, j) += params_w[f_0 * features + f].at(i, j);

                if (use_biases)
                {
                    //normal derivative
                    for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                        for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                            b_grad[f_0 * features + f].at(0, 0) += deriv[f_0].at(i_0, j_0);

                    //l2 weight decay
                    if (use_l2_weight_decay && include_biases_decay && online)
                        b_grad[f_0 * features + f].at(0, 0) += 2 * weight_decay_factor * params_b[f_0 * features + f].at(0, 0);
                }

                //update for online (momentum)
                if (use_momentum && online)
                {
                    for (size_t i = 0; i < kernel_size; ++i)
                    {
                        for (size_t j = 0; j < kernel_size; ++j)
                        {
                            params_w[f_0 * features + f].at(i, j) += -learning_rate * (w_grad[f_0 * features + f].at(i, j) + momentum_term * weights_momentum[f_0 * features + f].at(i, j));
                            weights_momentum[f_0 * features + f].at(i, j) = momentum_term * weights_momentum[f_0 * features + f].at(i, j) + w_grad[f_0 * features + f].at(i, j);
                            w_grad[f_0 * features + f].at(i, j) = 0;
                        }
                    }

                    if (use_biases)
                    {
                        params_b[f_0 * features + f].at(0, 0) += -learning_rate * (b_grad[f_0 * features + f].at(0, 0) + momentum_term * biases_momentum[f_0 * features + f].at(0, 0));
                        biases_momentum[f_0 * features + f].at(0, 0) = momentum_term * biases_momentum[f_0 * features + f].at(0, 0) + b_grad[f_0 * features + f].at(0, 0);
                        b_grad[f_0 * features + f].at(0, 0) = 0;
                    }
                }

                //vanilla online
                else if (online)
                {
                    for (size_t i = 0; i < kernel_size; ++i)
                    {
                        for (size_t j = 0; j < kernel_size; ++j)
                        {
                            params_w[f_0 * features + f].at(i, j) += -learning_rate * w_grad[f_0 * features + f].at(i, j);
                            w_grad[f_0 * features + f].at(i, j) = 0;
                        }
                    }

                    if (use_biases)
                    {
                        params_b[f_0 * features + f].at(0, 0) += -learning_rate * b_grad[f_0 * features + f].at(0, 0);
                        b_grad[f_0 * features + f].at(0, 0) = 0;
                    }
                }
            }
        }

        //apply derivatives (from chain rule)
        chain_activations(out_deriv, activations_pre, previous_layer_activation);
    }

    //batch feed forwards
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //batch feed backwards
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //batch backprop
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }

    //perform wake sleep DOESN'T ACCUMULATE IN GRADIENTS, applies directly
    static void wake_sleep(float& learning_rate, size_t markov_iterations, bool use_dropout)
    {
        constexpr size_t out_rows = use_padding ? rows / stride + 1 : (rows - kernel_size) / stride + 1;
        constexpr size_t out_cols = use_padding ? cols / stride + 1 : (cols - kernel_size) / stride + 1;

        //find difference via gibbs sampling
        feature_maps_type original = { 0 };
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
            stochastic_sample<features, rows, cols>(feature_maps);
        feed_forwards(reconstructed);
        for (size_t its = 1; its < markov_iterations; ++its)
        {
            stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
            feed_backwards(reconstructed);
            if (!mean_field)
                stochastic_sample<features, rows, cols>(feature_maps);
            feed_forwards(reconstructed);
        }

        constexpr size_t N = (kernel_size - 1) / 2;

        if (!mean_field)
            stochastic_sample<out_features, out_rows, out_cols>(discriminated);

        //adjust weights ONLINE
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
        if (use_biases && activation_function == MTNN_FUNC_RBM)
            for (size_t f = 0; f < features; ++f)
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        biases[f].at(i, j) += -learning_rate * (feature_maps[f].at(i, j) - original[f].at(i, j));
    }
};

//initialize static
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> bool ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::mean_field = false;
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<features, rows, cols> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights = { -.1f, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<((use_biases && activation_function == MTNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? cols : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::generative_biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<(use_biases ? out_features * features : 0), (use_biases ? 1 : 0), (use_biases ? 1 : 0)> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<out_features * features, kernel_size, kernel_size> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<0, 0, 0> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> FeatureMap<0, 0, 0> ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t kernel_size, size_t stride, size_t out_features, size_t activation_function, bool use_biases, bool use_padding> size_t ConvolutionLayer<index, features, rows, cols, kernel_size, stride, out_features, activation_function, use_biases, use_padding>::n = 0;

//A full connectivity layer. Note that the shape doesn't really matter wrt weights, biases (interprets as vector). Can use any activation function, biases or not
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> class PerceptronFullConnectivityLayer : public Layer_Functions<features, rows, cols>
{
public:

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    //biases (if used) are kept in own matrix
    static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases;
    //the weights. wij goes from node j to i
    static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights;
    //only used in wake-sleep/feed back/rbms
    static FeatureMap<((use_biases && activation_function == MTNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? cols : 0)> generative_biases;

    //used for hessian (old) or Adam
    static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_aux_data;
    //used for hessian (old) or Adam
    static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_aux_data;

    //stores actual gradient
    static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_gradient;
    //stores actual gradient
    static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_gradient;

    //stores momentum (if applicable)
    static FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> biases_momentum;
    //stores momentum (if applicable)
    static FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> weights_momentum;

    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_mean;
    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_PERCEPTRONFULLCONNECTIVITY;
    //activation function type (dynamic test, but not stored since constexpr)
    static constexpr size_t activation = activation_function;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<out_features, out_rows, out_cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //not used except batch norm
    static size_t n;
    //used in wake-sleep only
    static bool mean_field;

    //not used (static class)
    PerceptronFullConnectivityLayer() = default;

    //not used (static class)
    ~PerceptronFullConnectivityLayer() = default;

    //feed forwards given input, weights, biases to output
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
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
                                    params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j));

                    //add bias
                    if (use_biases)
                        output[f_0].at(i_0, j_0) = activate(sum + params_b[f_0].at(i_0, j_0), activation_function);
                    else
                        output[f_0].at(i_0, j_0) = activate(sum, activation_function);
                }
            }
        }
    }

    //undo feed forwards, with generative biases instead
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
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
                                sum += params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) * input[f_0].at(i_0, j_0);

                        if (use_biases && activation_function == MTNN_FUNC_RBM)
                            sum += params_b[f].at(i, j);
                        output[f].at(i, j) = activate(sum, activation_function);
                    }
                }
            }
        }
    }

    //accumulate gradients in given, using given weights, biases, outputs, activations, derivs, etc. WILL APPLY IF ONLINE LEARNING and vanilla backprop
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
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
                        b_grad[f_0].at(i_0, j_0) += deriv[f_0].at(i_0, j_0);

                        //L2 weight decay
                        if (use_l2_weight_decay && include_biases_decay && online)
                            b_grad[f_0].at(i_0, j_0) += 2 * weight_decay_factor * params_b[f_0].at(i_0, j_0);

                        //online update
                        if (use_momentum && online)
                        {
                            params_b[f_0].at(i_0, j_0) += -learning_rate * (b_grad[f_0].at(i_0, j_0) + momentum_term * biases_momentum[f_0].at(i_0, j_0));
                            biases_momentum[f_0].at(i_0, j_0) = momentum_term * biases_momentum[f_0].at(i_0, j_0) + b_grad[f_0].at(i_0, j_0);
                            b_grad[f_0].at(i_0, j_0) = 0;
                        }

                        else if (online)
                        {
                            params_b[f_0].at(i_0, j_0) += -learning_rate * b_grad[f_0].at(i_0, j_0);
                            b_grad[f_0].at(i_0, j_0) = 0;
                        }
                    }

                    for (size_t f = 0; f < features; ++f)
                    {
                        for (size_t i = 0; i < rows; ++i)
                        {
                            for (size_t j = 0; j < cols; ++j)
                            {
                                //update deltas
                                out_deriv[f].at(i, j) += deriv[f_0].at(i_0, j_0) * params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);

                                //normal derivative
                                w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += deriv[f_0].at(i_0, j_0) * activations_pre[f].at(i, j);

                                //L2 decay
                                if (use_l2_weight_decay && online)
                                    w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += 2 * weight_decay_factor * params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);

                                //Online updates
                                if (use_momentum && online)
                                {
                                    params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) += -learning_rate * (w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) + momentum_term * weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j));
                                    weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = momentum_term * weights_momentum[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) + w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);
                                    w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = 0;
                                }

                                else if (online)
                                {
                                    params_w[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) +=
                                        -learning_rate * w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j);
                                    w_grad[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, f * rows * cols + i * cols + j) = 0;
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

    //feed forwards batch
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //feed backwards batch
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //backprop batch
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }

    //perform wake sleep DOESN'T ACCUMULATE IN GRADIENTS, applies directly
    static void wake_sleep(float& learning_rate, size_t markov_iterations, bool use_dropout)
    {
        //find difference via gibbs sampling
        feature_maps_type original = { 0 };

        out_feature_maps_type discriminated = { 0 };

        out_feature_maps_type reconstructed = { 0 };

        //Sample, but don't "normalize" second time
        feed_forwards(feature_maps, discriminated);
        reconstructed = discriminated;
        stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
        feed_backwards(feature_maps, reconstructed);
        if (!mean_field)
            stochastic_sample<features, rows, cols>(feature_maps);
        feed_forwards(feature_maps, reconstructed);
        for (size_t its = 1; its < markov_iterations; ++its)
        {
            stochastic_sample<out_features, out_rows, out_cols>(reconstructed);
            feed_backwards(feature_maps, reconstructed);
            if (!mean_field)
                stochastic_sample<features, rows, cols>(feature_maps);
            feed_forwards(feature_maps, reconstructed);
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
                for (size_t i_0 = 0; i_0 < biases.rows(); ++i_0)
                    for (size_t j_0 = 0; j_0 < biases.cols(); ++j_0)
                        biases[f_0].at(i_0, j_0) += -learning_rate * (reconstructed[f_0].at(i_0, j_0) - discriminated[f_0].at(i_0, j_0));
        }

        //adjust visible biases
        if (use_biases && activation_function == MTNN_FUNC_RBM)
            for (size_t f = 0; f < features; ++f)
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        generative_biases[f].at(i, j) += -learning_rate * (feature_maps[f].at(i, j) - original[f].at(i, j));
    }
};

//static variable initialization
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> bool PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::mean_field = false;
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<features, rows, cols> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights = { -.1f, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<((use_biases && activation_function == MTNN_FUNC_RBM) ? features : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? rows : 0), ((use_biases && activation_function == MTNN_FUNC_RBM) ? cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::generative_biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<(use_biases ? out_features : 0), (use_biases ? out_rows : 0), (use_biases ? out_cols : 0)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<1, (out_features * out_rows * out_cols), (features * rows * cols)> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<0, 0, 0> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> FeatureMap<0, 0, 0> PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t activation_function, bool use_biases> size_t PerceptronFullConnectivityLayer<index, features, rows, cols, out_features, out_rows, out_cols, activation_function, use_biases>::n = 0;

//LSTM layer, max_t_store is the max number of steps to perform bptt on (may want to set to batch size)
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> class LSTMLayer : public Layer_Functions<features, rows, cols>
{

public:

    ////TODO: not having net storing (cell/hidden chains) may screw stuff up (specifically with parallel/weight updates)

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    ////4 feature maps for each seperate layer within the LSTM unit (forget, activation, influence, output)
    //biases are kept in own matrix
    static FeatureMap<4, out_features * out_rows * out_cols, 1> biases;
    //weights
    static FeatureMap<4, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols> weights;
    //only used in wake-sleep/feed back/rbms
    static FeatureMap<0, 0, 0> generative_biases;

    //used for hessian (old) or Adam
    static FeatureMap<4, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols> weights_aux_data;
    //used for hessian (old) or Adam
    static FeatureMap<4, out_features * out_rows * out_cols, 1> biases_aux_data;

    //stores actual gradient
    static FeatureMap<4, out_features * out_rows * out_cols, 1> biases_gradient;
    //stores actual gradient
    static FeatureMap<4, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols> weights_gradient;

    //stores momentum (if applicable)
    static FeatureMap<4, out_features * out_rows * out_cols, 1> biases_momentum;
    //stores momentum (if applicable)
    static FeatureMap<4, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols> weights_momentum;

    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_mean;
    //not used except batch norm
    static FeatureMap<0, 0, 0> activations_population_variance;

    ////internal lstm data; use for bptt since need to have all previous data. keep activation of all layers in vector, performs bptt on last one, push onto each feed forward (pop if full)
    static FeatureMapVector<out_features, out_rows, out_cols> cell_states; //keep max num of steps ONLY
    static FeatureMapVector<out_features, out_rows, out_cols> hidden_states;
    static FeatureMapVector<out_features, out_rows, out_cols> forget_states;
    static FeatureMapVector<out_features, out_rows, out_cols> influence_states;
    static FeatureMapVector<out_features, out_rows, out_cols> activation_states;
    static FeatureMapVector<out_features, out_rows, out_cols> output_states;

    //assumes that stores the derivs from next time step wrt cell state, needs to be reset after each batch update
    static FeatureMap<out_features, out_rows, out_cols> cell_state_deriv;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_LSTM;
    //just handle all of the chain rule in here, actual activations are significantly different
    static constexpr size_t activation = MTNN_FUNC_LINEAR; 

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<out_features, out_rows, out_cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //used since hidden plus inputs are given to each layer
    using concat_type = FeatureMap<1, out_features * out_rows * out_cols + features * rows * cols, 1>;

    //not used except batch norm
    static size_t n;

private:
    //combine hidden and inputs into one fm
    static inline FeatureMap<1, out_features * out_rows * out_cols + features * rows * cols, 1> concatenate(FeatureMap<features, rows, cols>& a, FeatureMap<out_features, out_rows, out_cols>& b)
    {
        //do hidden first
        FeatureMap<1, out_features * out_rows * out_cols + features * rows * cols, 1> out = {};
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                    out[0].at(f_0 * out_rows * out_cols + i_0 * out_cols + j_0, 0) = b[f_0].at(i_0, j_0);

        //then inputs
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    out[0].at(out_features * out_rows * out_cols + f * rows * cols + i * cols + j, 0) = a[f].at(i, j);
        return out;
    }

    //These methods are basic feed forward methods, re-implemented here for parallel/convenience (only use 1 FM instead of a vector)
    static void feed_forwards_gate(size_t activation, FeatureMap<1, out_features * out_rows * out_cols + features * rows * cols, 1>& input, FeatureMap<out_features, out_rows, out_cols>& output, Matrix2D<float, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols>& params_w, Matrix2D<float, out_features * out_rows * out_cols, 1>& params_b)
    {
        //loop all outputs
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
            {
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                {
                    auto& targ = output[f_0].at(i_0, j_0);
                    size_t idx = f_0 * out_rows * out_cols + i_0 * out_cols + j_0;
                    //loop all inputs
                    for (size_t i = 0; i < out_features * out_rows * out_cols + features * rows * cols; ++i)
                        targ += input[0].at(i, 0) * params_w.at(idx, i);
                    targ = activate(targ + params_b.at(idx, 0), activation); //add bias and activate
                }
            }
        }
    }

    //backprop for only one fully connected gate
    static void back_prop_gate(size_t activation, FeatureMap<out_features, out_rows, out_cols>& d_hidden, FeatureMap<out_features, out_rows, out_cols>& outputs_pre, FeatureMap<1, out_features * out_rows * out_cols + features * rows * cols, 1>& activations_pre, FeatureMap<features, rows, cols>& out_derivs, Matrix2D<float, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols>& params_w, Matrix2D<float, out_features * out_rows * out_cols, 1>& params_b, Matrix2D<float, out_features * out_rows * out_cols, out_features * out_rows * out_cols + features * rows * cols>& params_w_grad, Matrix2D<float, out_features * out_rows * out_cols, 1>& params_b_grad)
    {
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
            {
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                {
                    auto& deri = d_hidden[f_0].at(i_0, j_0);
                    deri *= activation_derivative(outputs_pre[f_0].at(i_0, j_0), activation);
                    size_t idx = f_0 * out_rows * out_cols + i_0 * out_cols + j_0;

                    //UPDATE ALL PARAMS

                    params_b_grad.at(idx, 0) += deri;
                    //loop all weights
                    for (size_t f2 = 0; f2 < out_features; ++f2)
                    {
                        for (size_t i2 = 0; i2 < out_rows; ++i2)
                        {
                            for (size_t j2 = 0; j2 < out_cols; ++j2)
                            {
                                size_t idx2 = f2 * out_rows * out_cols + i2 * out_cols + j2;
                                //out_derivs[f].at(i, j) += deri * params_w.at(idx, idx2); todo doesn't backprop to next hidden?
                                params_w_grad.at(idx, idx2) += deri * activations_pre[0].at(idx, idx2);
                            }
                        }
                    }
                    for (size_t f = 0; f < features; ++f)
                    {
                        for (size_t i = 0; i < rows; ++i)
                        {
                            for (size_t j = 0; j < cols; ++j)
                            {
                                //offset
                                size_t idx2 = out_features * out_rows * out_cols + f * rows * cols + i * cols + j;
                                out_derivs[f].at(i, j) += deri * params_w.at(idx, idx2);
                                params_w_grad.at(idx, idx2) += deri * activations_pre[0].at(idx, idx2);
                            }
                        }
                    }
                }
            }
        }
    }

    //perform bptt once for a given idx (will only call once from back_prop)
    static void back_prop_through_time(size_t idx, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, weights_type& params_w, biases_type& params_b, weights_type& w_grad, biases_type& b_grad)
    {
        //calculate derivs wrt each layer
        out_feature_maps_type d_out = {};
        out_feature_maps_type d_activation = {};
        out_feature_maps_type d_influence = {};
        out_feature_maps_type d_forget = {};
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
            {
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                {
                    //update cell state deriv
                    cell_state_deriv[f_0].at(i_0, j_0) += deriv[f_0].at(i_0, j_0) * output_states[idx][f_0].at(i_0, j_0) * activation_derivative(cell_states[idx][f_0].at(i_0, j_0), MTNN_FUNC_TANH);

                    //update the rest
                    auto& d_c_t = cell_state_deriv[f_0].at(i_0, j_0);
                    d_out[f_0].at(i_0, j_0) = deriv[f_0].at(i_0, j_0) * tanh(cell_states[idx][f_0].at(i_0, j_0));
                    d_activation[f_0].at(i_0, j_0) = d_c_t * influence_states[idx][f_0].at(i_0, j_0);
                    d_influence[f_0].at(i_0, j_0) = d_c_t * activation_states[idx][f_0].at(i_0, j_0);
                    d_forget[f_0].at(i_0, j_0) = d_c_t * cell_states[idx - 1][f_0].at(i_0, j_0);
                    cell_state_deriv[f_0].at(i_0, j_0) *= forget_states[idx][f_0].at(i_0, j_0); //update last time
                }
            }
        }

        auto input = concatenate(activations_pre, hidden_states[idx]);

        //pass to all layers and compute
        back_prop_gate(MTNN_FUNC_LOGISTIC, d_forget, forget_states[idx], input, out_deriv, params_w[0], params_b[0], w_grad[0], b_grad[0]);
        back_prop_gate(MTNN_FUNC_LOGISTIC, d_influence, influence_states[idx], input, out_deriv, params_w[1], params_b[1], w_grad[1], b_grad[1]);
        back_prop_gate(MTNN_FUNC_TANH, d_activation, activation_states[idx], input, out_deriv, params_w[2], params_b[2], w_grad[2], b_grad[2]);
        back_prop_gate(MTNN_FUNC_LOGISTIC, d_out, output_states[idx], input, out_deriv, params_w[3], params_b[3], w_grad[3], b_grad[3]);
    }

public:

    //don't use, static class
    LSTMLayer() = default;

    //don't us, static class
    ~LSTMLayer() = default;


    //feed forwards given input, weights, biases to output
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        //create real input to each gate
        auto input_cat = concatenate(input, hidden_states.back());

        //create a new target for output
        cell_states.push_back(out_feature_maps_type{});
        hidden_states.push_back(out_feature_maps_type{});
        forget_states.push_back(out_feature_maps_type{});
        influence_states.push_back(out_feature_maps_type{});
        activation_states.push_back(out_feature_maps_type{});
        output_states.push_back(out_feature_maps_type{});

        size_t idx = cell_states.size() - 1;

        //forget gate feed forward
        feed_forwards_gate(MTNN_FUNC_LOGISTIC, input_cat, forget_states[idx], params_w[0], params_b[0]);
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                    cell_states[idx][f_0].at(i_0, j_0) = cell_states[idx - 1][f_0].at(i_0, j_0) * forget_states[idx][f_0].at(i_0, j_0);
        
        //get current activations and combine with forgotten
        feed_forwards_gate(MTNN_FUNC_LOGISTIC, input_cat, influence_states[idx], params_w[1], params_b[1]);
        feed_forwards_gate(MTNN_FUNC_TANH, input_cat, activation_states[idx], params_w[2], params_b[2]);
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                    cell_states[idx][f_0].at(i_0, j_0) += activation_states[idx][f_0].at(i_0, j_0) * influence_states[idx][f_0].at(i_0, j_0);

        //get hidden/output
        feed_forwards_gate(MTNN_FUNC_LOGISTIC, input_cat, output_states[idx], params_w[3], params_b[3]);
        for (size_t f_0 = 0; f_0 < out_features; ++f_0)
        {
            for (size_t i_0 = 0; i_0 < out_rows; ++i_0)
            {
                for (size_t j_0 = 0; j_0 < out_cols; ++j_0)
                {
                    hidden_states[idx][f_0].at(i_0, j_0) = output_states[idx][f_0].at(i_0, j_0) * tanh(cell_states[idx][f_0].at(i_0, j_0));
                    output[f_0].at(i_0, j_0) = hidden_states[idx][f_0].at(i_0, j_0);
                }
            }
        }

        //adjust stored sizes if necessary
        if (idx >= max_t_store + 1)
        {
            hidden_states.erase(hidden_states.begin());
            cell_states.erase(cell_states.begin());
            forget_states.erase(forget_states.begin());
            influence_states.erase(influence_states.begin());
            activation_states.erase(activation_states.begin());
            output_states.erase(output_states.begin());
        }
    }

    //undo feed forwards, with generative biases instead TODO implement
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        //todo: update to real
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = input[f].at(i, j);
    }

    //accumulate gradients in given, using given weights, biases, outputs, activations, derivs, etc. WON'T APPLY IF ONLINE (TODO)
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        size_t idx = cell_states.size() - 1;//todo training doesn't make sense with only one? (unless ordered/popped correctly)

        back_prop_through_time(idx, deriv, activations_pre, out_deriv, params_w, params_b, w_grad, b_grad);
        
        //apply derivatives
        chain_activations(out_deriv, activations_pre, previous_layer_activation);
    }

    //feed forwards batch - very useful for bptt
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases, bool discriminating = false)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //feed back batch
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //backprop batch. Is interpreted as a time forward, so will perform bptt on whole batch (assumes that max_t_store >= batch size)
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        //zero
        cell_state_deriv = { 0 };
        for (size_t in = derivs.size() - 1; in > 0; --in)
        {
            //backprop
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);

            //update last for correct
            hidden_states.pop_back();
            cell_states.pop_back();
            forget_states.pop_back();
            influence_states.pop_back();
            activation_states.pop_back();
            output_states.pop_back();
        }
    }
};

//static variable initialization
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<features, rows, cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::feature_maps = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, out_features * out_rows * out_cols, 1> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, (out_features * out_rows * out_cols), (features * rows * cols + out_features * out_rows * out_cols)> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::weights = { -.1f, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<0, 0, 0> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::generative_biases = { 0, .1f };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, (out_features * out_rows * out_cols), (features * rows * cols + out_features * out_rows * out_cols)> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::weights_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, out_features * out_rows * out_cols, 1> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::biases_aux_data = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, out_features * out_rows * out_cols, 1> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::biases_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, (out_features * out_rows * out_cols), (features * rows * cols + out_features * out_rows * out_cols)> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::weights_gradient = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, out_features * out_rows * out_cols, 1> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::biases_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<4, (out_features * out_rows * out_cols), (features * rows * cols + out_features * out_rows * out_cols)> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::weights_momentum = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<0, 0, 0> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::activations_population_mean = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<0, 0, 0> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::activations_population_variance = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> size_t LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::n = 0;

//class specific stuff
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMap<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::cell_state_deriv = { 0 }; //keep max num of steps ONLY
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::cell_states = { 0 }; //keep max num of steps ONLY
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::hidden_states = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::forget_states = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::influence_states = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::activation_states = { 0 };
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_features, size_t out_rows, size_t out_cols, size_t max_t_store> FeatureMapVector<out_features, out_rows, out_cols> LSTMLayer<index, features, rows, cols, out_features, out_rows, out_cols, max_t_store>::output_states = { 0 };

//Batch normalization uses population stats for feed forward, calculates batch statistics, CANNOT TRAIN WITHOUT BATCH TRAINING, activation is applied after statistical transformation
template<size_t index, size_t features, size_t rows, size_t cols, size_t activation_function> class BatchNormalizationLayer : public Layer_Functions<features, rows, cols>
{
public:
    //TODO updates in parallel, can't use with Adam (minibatch statistics?)

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;

    //use for discrim after training (or online during training)
    static FeatureMap<features, rows, cols> activations_population_mean;
    //use for discrim after training (or online during training)
    static FeatureMap<features, rows, cols> activations_population_variance;

    //beta
    static FeatureMap<features, rows, cols> biases;
    //gamma
    static FeatureMap<features, rows, cols> weights;

    //stores actual gradient
    static FeatureMap<features, rows, cols> biases_gradient;
    //stores actual gradient
    static FeatureMap<features, rows, cols> weights_gradient;

    //aren't used in batch norm?
    static FeatureMap<0, 0, 0> generative_biases;
    //used for hessian (old) or Adam
    static FeatureMap<features, rows, cols> biases_aux_data;
    //used for hessian (old) or Adam
    static FeatureMap<features, rows, cols> weights_aux_data;

    //stores momentum (if applicable)
    static FeatureMap<features, rows, cols> biases_momentum;
    //stores momentum (if applicable)
    static FeatureMap<features, rows, cols> weights_momentum;

    //is used just to prevent division by zero
    static const float min_divisor;
    //keeps track of batch size (for population)
    static size_t n;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_BATCHNORMALIZATION;
    //activation function type (dynamic test, but not stored since constexpr)
    static constexpr size_t activation = activation_function;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<features, rows, cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //won't use, static class
    BatchNormalizationLayer() = default;

    //won't use, static class
    ~BatchNormalizationLayer() = default;

    //feed forwards given input, weights, biases to output; uses population
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = activate(params_w[f].at(i, j) * (input[f].at(i, j) - activations_population_mean[f].at(i, j)) / sqrt(activations_population_variance[f].at(i, j) + min_divisor) + params_b[f].at(i, j), activation_function);
    }

    //undo feed forwards, currently just does feed forwards though
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)//todo?
                    output[f].at(i, j) = params_w[f].at(i, j) * (input[f].at(i, j) - activations_population_mean[f].at(i, j)) / sqrt(activations_population_variance[f].at(i, j) + min_divisor) + params_b[f].at(i, j);
    }

    //accumulate gradients in given, using given weights, biases, outputs, activations, derivs, etc. WILL APPLY IF ONLINE LEARNING and vanilla backprop
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        //undefined for single 
    }

    //feed forwards for batch and uses the batch's statistics (not population) and then updates population statistics
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases, bool discriminating = false)
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
                    float gamma = params_w[f].at(i, j);
                    float beta = params_b[f].at(i, j);
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

    //batch feed backwards
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //ONLY WAY TO TRAIN uses minibatch statistics (which are stored in aux_data)
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
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
                        float mu = biases_aux_data[f].at(i, j);
                        float div = activations_pre[f].at(i, j) - mu;
                        float std = sqrt(weights_aux_data[f].at(i, j) + min_divisor);

                        float xhat = div / std;
                        float d_out = deriv[f].at(i, j);

                        b_grad[f].at(i, j) += d_out;
                        w_grad[f].at(i, j) += d_out * xhat;

                        float sumDeriv = 0.0f;
                        float sumDiff = 0.0f;
                        for (size_t in2 = 0; in2 < out_derivs.size(); ++in2)
                        {
                            float d_outj = out_derivs[in2][f].at(i, j);
                            sumDeriv += d_outj;
                            sumDiff += d_outj * (activations_pre_vec[in2][f].at(i, j) - mu);
                        }

                        out_deriv[f].at(i, j) = params_w[f].at(i, j) / derivs.size() / std * (derivs.size() * d_out - sumDeriv - div / std * sumDiff);
                    }
                }
            }

            //apply derivatives
            chain_activations(out_deriv, activations_pre, previous_layer_activation);
        }
    }
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

//This pooling layer takes the maximum values in a region and puts in the output
template<size_t index, size_t features, size_t rows, size_t cols, size_t out_rows, size_t out_cols> class MaxpoolLayer : public Layer_Functions<features, rows, cols>
{
public:
    //todo storing doesn't work in parallel

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    //no parameters
    static FeatureMap<0, 0, 0> biases;
    //no parameters
    static FeatureMap<0, 0, 0> weights;
    //no parameters
    static FeatureMap<0, 0, 0> generative_biases;

    //no parameters
    static FeatureMap<0, 0, 0> weights_aux_data;
    //no parameters
    static FeatureMap<0, 0, 0> biases_aux_data;

    //no parameters
    static FeatureMap<0, 0, 0> biases_gradient;
    //no parameters
    static FeatureMap<0, 0, 0> weights_gradient;

    //no parameters
    static FeatureMap<0, 0, 0> biases_momentum;
    //no parameters
    static FeatureMap<0, 0, 0> weights_momentum;

    //no parameters
    static FeatureMap<0, 0, 0> activations_population_mean;
    //no parameters
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_MAXPOOL;
    //no activation function
    static constexpr size_t activation = MTNN_FUNC_LINEAR;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<features, out_rows, out_cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //not used except in batch norm
    static size_t n;

    //won't use, static class
    MaxpoolLayer() = default;

    //won't use, static class
    ~MaxpoolLayer() = default;

    //feed forwards given input, weights, biases to output
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        //set minimum as negative
        for (size_t f_0 = 0; f_0 < features; ++f_0)
            for (size_t i = 0; i < output[f_0].rows(); ++i)
                for (size_t j = 0; j < output[f_0].cols(); ++j)
                    output[f_0].at(i, j) = -INFINITY;

        for (size_t f = 0; f < features; ++f)
        {
            //get size of region
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

    //backprops values to previous layer
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
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
                            //prop deriv to strongest cell
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

    //backprops derivatives to next layer (no parameters so no update)
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        //TODO fails on batch

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
                    //backprop deriv if it was maximum
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

    //batch feed forwards
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //batch feed backwards
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //batch train
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }

private:
    //used to keep track of which cell was maximum in each region
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

//Transforms output according to softmax function
template<size_t index, size_t features, size_t rows, size_t cols> class SoftMaxLayer : public Layer_Functions<features, rows, cols>
{
public:

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;

    //no parameters
    static FeatureMap<0, 0, 0> biases;
    //no parameters
    static FeatureMap<0, 0, 0> weights;
    //no parameters
    static FeatureMap<0, 0, 0> generative_biases;

    //no parameters
    static FeatureMap<0, 0, 0> weights_aux_data;
    //no parameters
    static FeatureMap<0, 0, 0> biases_aux_data;

    //no parameters
    static FeatureMap<0, 0, 0> biases_gradient;
    //no parameters
    static FeatureMap<0, 0, 0> weights_gradient;

    //no parameters
    static FeatureMap<0, 0, 0> biases_momentum;
    //no parameters
    static FeatureMap<0, 0, 0> weights_momentum;

    //no parameters
    static FeatureMap<0, 0, 0> activations_population_mean;
    //no parameters
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_SOFTMAX;
    //no activation funciton
    static constexpr size_t activation = MTNN_FUNC_LINEAR;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<features, rows, cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //only used for batch norm
    static size_t n;

    //won't use, static class
    SoftMaxLayer() = default;

    //won't use, static class
    ~SoftMaxLayer() = default;

    //compute the softmax
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t f = 0; f < features; ++f)
        {
            //find total
            float sum = 0.0f;
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    sum += input[f].at(i, j) < 6 ? exp(input[f].at(i, j)) : exp(6);

            //get prob
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = (input[f].at(i, j) < 6 ? exp(input[f].at(i, j)) : exp(6)) / sum;
        }
    }

    //undo, makes assumptions about total value
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        //assume that the original input has a mean of 0, so sum of original input would *approximately* be the total number of inputs
        size_t total = rows * cols;

        //inverse
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = log(total * input[f].at(i, j));
    }

    //backprops derivs, no update
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        //calculate sum of all derivs
        std::vector<float> sums(features);
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    sums[f] += deriv[f].at(i, j);

        //recalculate activations for next layer (little inefficient, but oh well)
        out_feature_maps_type out_vals{};
        feed_forwards(activations_pre, out_vals, params_w, params_b);

        //compute derivative as = out_act * sum_derivs - deriv
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    out_deriv[f].at(i, j) = out_vals[f].at(i, j) * sums[f] - deriv[f].at(i, j);

        //apply derivatives
        chain_activations(out_deriv, activations_pre, previous_layer_activation);
    }

    //feed forwards batch
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases, bool discriminating = false)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //feed backwards batch
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //backprop batch
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }
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

//Basic input layer, should only be at beginning, works in middle but doesn't make sense
template<size_t index, size_t features, size_t rows, size_t cols> class InputLayer : public Layer_Functions<features, rows, cols>
{
public:

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    //no parameters
    static FeatureMap<0, 0, 0> biases;
    //no parameters
    static FeatureMap<0, 0, 0> weights;
    //no parameters
    static FeatureMap<0, 0, 0> generative_biases;

    //no parameters
    static FeatureMap<0, 0, 0> weights_aux_data;
    //no parameters
    static FeatureMap<0, 0, 0> biases_aux_data;

    //no parameters
    static FeatureMap<0, 0, 0> biases_gradient;
    //no parameters
    static FeatureMap<0, 0, 0> weights_gradient;

    //no parameters
    static FeatureMap<0, 0, 0> biases_momentum;
    //no parameters
    static FeatureMap<0, 0, 0> weights_momentum;

    //no parameters
    static FeatureMap<0, 0, 0> activations_population_mean;
    //no parameters
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_INPUT;
    //no transformation
    static constexpr size_t activation = MTNN_FUNC_LINEAR;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<features, rows, cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //won't use except in batch norm
    static size_t n;

    //won't use, static class
    InputLayer() = default;

    //won't use, static class
    ~InputLayer() = default;

    //basic copy
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        //just output
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = input[f].at(i, j);
    }

    //basic copy
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        //just output
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = input[f].at(i, j);
    }

    //basic copy
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    out_deriv[f].at(i, j) = deriv[f].at(i, j);
        //apply derivatives
        chain_activations(out_deriv, activations_pre, previous_layer_activation);
    }

    //batch copy
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases, bool discriminating = false)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //batch copy
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //batch copy
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }
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

    //feature maps - DO NOT STORE activations if fed forwards - not used for much
    static FeatureMap<features, rows, cols> feature_maps;
    //no parameters
    static FeatureMap<0, 0, 0> biases;
    //no parameters
    static FeatureMap<0, 0, 0> weights;
    //no parameters
    static FeatureMap<0, 0, 0> generative_biases;

    //no parameters
    static FeatureMap<0, 0, 0> weights_aux_data;
    //no parameters
    static FeatureMap<0, 0, 0> biases_aux_data;

    //no parameters
    static FeatureMap<0, 0, 0> biases_gradient;
    //no parameters
    static FeatureMap<0, 0, 0> weights_gradient;

    //no parameters
    static FeatureMap<0, 0, 0> biases_momentum;
    //no parameters
    static FeatureMap<0, 0, 0> weights_momentum;

    //no parameters
    static FeatureMap<0, 0, 0> activations_population_mean;
    //no parameters
    static FeatureMap<0, 0, 0> activations_population_variance;

    //type of layer (dynamic test, but not stored since constexpr)
    static constexpr size_t type = MTNN_LAYER_OUTPUT;
    //no transformation
    static constexpr size_t activation = MTNN_FUNC_LINEAR;

    //define for creating tuples or using within templates
    using out_feature_maps_type = FeatureMap<features, rows, cols>;
    using weights_type = decltype(weights);
    using biases_type = decltype(biases);
    using generative_biases_type = decltype(generative_biases);
    using out_feature_maps_vector_type = std::vector<out_feature_maps_type>;
    using weights_vector_type = std::vector<weights_type>;
    using biases_vector_type = std::vector<biases_type>;
    using generative_biases_vector_type = std::vector<generative_biases_type>;

    //won't use outside of batch norm
    static size_t n;

    //won't use, static class
    OutputLayer() = default;

    //won't use, static class
    ~OutputLayer() = default;

    //basic copy
    static void feed_forwards(feature_maps_type& input, out_feature_maps_type& output, weights_type& params_w = weights, biases_type& params_b = biases)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = input[f].at(i, j);
    }

    //basic copy
    static void feed_backwards(feature_maps_type& output, out_feature_maps_type& input, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    output[f].at(i, j) = input[f].at(i, j);
    }

    //basic copy
    static void back_prop(size_t previous_layer_activation, out_feature_maps_type& deriv, feature_maps_type& activations_pre, feature_maps_type& out_deriv, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t f = 0; f < features; ++f)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    out_deriv[f].at(i, j) = deriv[f].at(i, j);
        //apply derivatives
        chain_activations(out_deriv, activations_pre, previous_layer_activation);
    }

    //batch copy
    static void feed_forwards(feature_maps_vector_type& inputs, out_feature_maps_vector_type& outputs, weights_type& params_w = weights, biases_type& params_b = biases, bool discriminating = false)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_forwards(inputs[in], outputs[in], params_w, params_b);
    }

    //batch copy
    static void feed_backwards(feature_maps_vector_type& outputs, out_feature_maps_vector_type& inputs, weights_type& params_w = weights, generative_biases_type& params_b = generative_biases)
    {
        for (size_t in = 0; in < outputs.size(); ++in)
            feed_backwards(outputs[in], inputs[in], params_w, params_b);
    }

    //batch copy
    static void back_prop(size_t previous_layer_activation, out_feature_maps_vector_type& derivs, feature_maps_vector_type& activations_pre_vec, feature_maps_vector_type& out_derivs, bool online, float learning_rate, bool use_momentum, float momentum_term, bool use_l2_weight_decay, bool include_biases_decay, float weight_decay_factor, weights_type& params_w = weights, biases_type& params_b = biases, weights_type& w_grad = weights_gradient, biases_type& b_grad = biases_gradient)
    {
        for (size_t in = 0; in < derivs.size(); ++in)
            back_prop(previous_layer_activation, derivs[in], activations_pre_vec[in], out_derivs[in], false, learning_rate, use_momentum, momentum_term, use_l2_weight_decay, include_biases_decay, weight_decay_factor, params_w, params_b, w_grad, b_grad);
    }
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