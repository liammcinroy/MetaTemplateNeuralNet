#pragma once

#include <cstring>
#include <stdio.h>
#include <vector>

#include "imatrix.h"
#include "ilayer.h"

//default, MSE
#define CNN_LOSS_SQUAREERROR 0
//assumes prior layer is softmax
#define CNN_LOSS_LOGLIKELIHOOD 1
//undefined for error, instead sets output to labels during training
#define CNN_LOSS_CUSTOMTARGETS 2

//vanilla, add in momentum or hessian if desired
#define CNN_OPT_BACKPROP 0
//can't use with momentum or hessian
#define CNN_OPT_ADAM 1
//can't use with momentum or hessian
#define CNN_OPT_ADAGRAD 2

//TEMPLATE FOR LOOP

namespace std
{
	template< bool B, class T = void >
	using enable_if_t = typename std::enable_if<B, T>::type;
}

template<size_t i, size_t UPPER, size_t STEP, template<size_t> class func> struct for_loop_inc_impl
{
	template<size_t i2 = i>
	for_loop_inc_impl(std::enable_if_t<(i2 < UPPER), for_loop_inc_impl<i2, UPPER, STEP, func>>* = 0)
	{
		func<i>();
		auto next = for_loop_inc_impl<i + STEP, UPPER, STEP, func>();
	}

	template<size_t i2 = i>
	for_loop_inc_impl(std::enable_if_t<i2 == UPPER, for_loop_inc_impl<i2, UPPER, STEP, func>>* = 0)
	{
		func<i>();
	}
};

template<size_t i, size_t LOWER, size_t STEP, template<size_t> class func> struct for_loop_dec_impl
{
	template<size_t i2 = i>
	for_loop_dec_impl(std::enable_if_t<(i2 > LOWER), for_loop_dec_impl<i2, LOWER, STEP, func>>* = 0)
	{
		func<i>();
		auto next = for_loop_dec_impl<i - STEP, LOWER, STEP, func>();
	}

	template<size_t i2 = i>
	for_loop_dec_impl(std::enable_if_t<i2 == LOWER, for_loop_dec_impl<i2, LOWER, STEP, func>>* = 0)
	{
		func<i>();
	}
};

template<size_t START, size_t FINISH, size_t STEP, template<size_t> class func> struct for_loop
{
	template<size_t START2 = START>
	for_loop(std::enable_if_t<(START2 < FINISH), for_loop<START2, FINISH, STEP, func>>* = 0)
	{
		for_loop_inc_impl<START, FINISH, STEP, func>();
	}

	template<size_t START2 = START>
	for_loop(std::enable_if_t<(START2 > FINISH), for_loop<START2, FINISH, STEP, func>>* = 0)
	{
		for_loop_dec_impl<START, FINISH, STEP, func>();
	}

	template<size_t START2 = START>
	for_loop(std::enable_if_t<(START2 == FINISH), for_loop<START2, FINISH, STEP, func>>* = 0)
	{
		func<START>();
	}
};

//RECURSIVE PACK GET

template<size_t N, typename T0, typename... Ts> struct get_type_impl
{
	using type = typename get_type_impl<N - 1, Ts...>::type;
};

template<typename T0, typename... Ts>
struct get_type_impl<0, T0, Ts...>
{
	using type = T0;
};

template<size_t N, typename... Ts> using get_type = typename get_type_impl<N, Ts...>::type;

//RECURSIVE RBM INDEX GET

template<size_t n, typename... Ts> struct get_rbm_idx_impl
{
	static constexpr size_t idx = (get_type<n, Ts...>::activation == CNN_FUNC_RBM) ? n : get_rbm_idx_impl<n - 1, Ts...>::idx;
};

template<typename... Ts> struct get_rbm_idx_impl<0, Ts...>
{
	static constexpr size_t idx = 0;
};

template<typename... Ts> using get_rbm_idx = get_rbm_idx_impl<sizeof...(Ts)-1, Ts...>;

//CONSTEXPR STRING

template<size_t N, template<size_t...> class func, size_t... indices> struct do_foreach_range
{
	using type = typename do_foreach_range<N - 1, func, N - 1, indices...>::type;
};

template<template<size_t...> class func, size_t... indices> struct do_foreach_range<0, func, indices...>
{
	using type = typename func<indices...>::type;
};

template<char... cs> struct str
{
	static constexpr const char string[sizeof...(cs)+1] = { cs..., '\0' };
};

template<char... cs> constexpr const char str<cs...>::string[];

template<typename str_type> struct builder//str_type is static class with string literal
{
	template<size_t... indices> struct do_foreach//will be func
	{
		//want to fetch the char of each index
		using type = str<str_type{}.chars[indices]...>;
	};
};

#define CSTRING(string_literal) []{ \
    struct const_str { const char* chars = string_literal; }; \
    return do_foreach_range<sizeof(string_literal) - 1, builder<const_str>::do_foreach>::type{}; }()

template<size_t loss_function = CNN_LOSS_SQUAREERROR, size_t optimization_method = CNN_OPT_BACKPROP, bool use_dropout = false, bool use_batch_learning = false, bool use_momentum = false, bool use_hessian = false, bool use_l2_weight_decay = false, bool include_bias_decay = false, bool use_batch_normalization = false, bool keep_running_activation_statistics = false, bool collect_data_while_training = false, typename... layers>
class NeuralNet
{
private:

	//LAYER LOOP BODIES

	template<typename file_name> struct save_data_t
	{
	public:
		save_data_t()
		{
#ifdef _MSC_VER
			fopen_s(&fp, file_name::string, "w+b");
#else
			fp = fopen(file_name::string, "w+b");
#endif
			loop_up_layers<save_data_impl>();
			fclose(fp);
		}
	private:
		static FILE* fp;

		template<size_t l> struct save_data_impl
		{
		private:
			void write_float(char bin[sizeof(float)], const float& f, FILE* file)
			{
				std::memcpy(bin, &f, sizeof(float));
				fputs(bin, file);
			}
		public:
			save_data_impl()
			{
				using layer = get_layer<l>;

				char bin[sizeof(float)]{};//for putting in binary data

										  //begin weights values
				{
					using t = decltype(layer::weights);
					for (size_t d = 0; d < t::size(); ++d)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								write_float(bin, layer::weights[d].at(i, j), fp);
				}

				//begin biases values
				{
					using t = decltype(layer::biases);
					for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
						for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
							for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
								write_float(bin, layer::biases[f_0].at(i_0, j_0), fp);//bias values
				}

				//begin gen biases values
				{
					using t = decltype(layer::generative_biases);
					for (size_t f = 0; f < t::size(); ++f)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								write_float(bin, layer::generative_biases[f].at(i, j), fp);//gen bias values
				}
			}
		};
	};

	template<typename file_name> struct load_data_t
	{
	public:
		load_data_t()
		{
#ifdef _MSC_VER
			fopen_s(&fp, file_name::string, "r+b");
#else
			fp = fopen(file_name::string, "r+b");
#endif
			loop_up_layers<load_data_impl>();
			fclose(fp);
		}
	private:
		static FILE* fp;

		template<size_t l> struct load_data_impl
		{
		private:
			void read_float(char bin[sizeof(float)], float& out_float, FILE* file)
			{
				fgets(bin, sizeof(float) + 1, file);
				std::memcpy(&out_float, bin, sizeof(float));
			}
		public:
			load_data_impl()
			{
				using layer = get_layer<l>;

				char bin[sizeof(float) + 1]{};

				//begin weights values
				{
					using t = decltype(layer::weights);
					for (size_t d = 0; d < t::size(); ++d)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								read_float(bin, layer::weights[d].at(i, j), fp);
				}

				//begin biases values
				{
					using t = decltype(layer::biases);
					for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
						for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
							for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
								read_float(bin, layer::biases[f_0].at(i_0, j_0), fp);
				}

				//begin gen biases values
				{
					using t = decltype(layer::generative_biases);
					for (size_t f = 0; f < t::size(); ++f)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								read_float(bin, layer::generative_biases[f].at(i, j), fp);
				}
			}
		};
	};

	template<size_t l, size_t target> struct reset_impl
	{
	public:
		reset_impl()
		{
			using layer = get_layer<l>;
			if (target == CNN_DATA_FEATURE_MAP)
			{
				using t = decltype(layer::feature_maps);
				for (size_t f = 0; f < t::size(); ++f)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::feature_maps[f].at(i, j) = 0.0f;
			}
			if (target == CNN_DATA_WEIGHT_GRAD)
			{
				using t = decltype(layer::weights_gradient);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_gradient[d].at(i, j) = 0.0f;
			}
			if (target == CNN_DATA_BIAS_GRAD)
			{
				using t = decltype(layer::biases_gradient);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) = 0.0f;
			}
			if (target == CNN_DATA_WEIGHT_MOMENT)
			{
				using t = decltype(layer::weights_momentum);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_momentum[d].at(i, j) = 0.0f;
			}
			if (target == CNN_DATA_BIAS_MOMENT)
			{
				using t = decltype(layer::biases_gradient);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) = 0.0f;
			}
			if (target == CNN_DATA_WEIGHT_HESSIAN)
			{
				using t = decltype(layer::weights_hessian);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_hessian[d].at(i, j) = 0.0f;
			}
			if (target == CNN_DATA_BIAS_HESSIAN)
			{
				using t = decltype(layer::biases_hessian);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_hessian[f_0].at(i_0, j_0) = 0.0f;
			}
			if (target == CNN_DATA_ACT)
			{
				using t = decltype(layer::activations_mean);
				for (size_t f = 0; f < t::size(); ++f)
				{
					for (size_t i = 0; i < t::rows(); ++i)
					{
						for (size_t j = 0; j < t::cols(); ++j)
						{
							layer::activations_mean[f].at(i, j) = 0.0f;
							layer::activations_variance[f].at(i, j) = 0.0f;
						}
					}
				}
			}
		}
	};

	template<size_t l, size_t target> struct delete_impl
	{
	public:
		delete_impl()
		{
			using layer = get_layer<l>;
			if (target == CNN_DATA_FEATURE_MAP)
			{
				using t = decltype(layer::feature_maps);
				layer::feature_maps.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
			if (target == CNN_DATA_WEIGHT_MOMENT)
			{
				using t = decltype(layer::weights_momentum);
				layer::weights_momentum.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
			if (target == CNN_DATA_BIAS_MOMENT)
			{
				using t = decltype(layer::biases_momentum);
				layer::biases_momentum.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
			if (target == CNN_DATA_WEIGHT_HESSIAN)
			{
				using t = decltype(layer::weights_hessian);
				layer::weights_hessian.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
			if (target == CNN_DATA_BIAS_HESSIAN)
			{
				using t = decltype(layer::biases_hessian);
				layer::biases_hessian.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
			if (target == CNN_DATA_ACT)
			{
				using t = decltype(layer::activations_mean);
				layer::activations_mean.~FeatureMaps<t::size(), t::rows(), t::cols()>();
				layer::activations_variance.~FeatureMaps<t::size(), t::rows(), t::cols()>();
			}
		}
	};

	template<size_t l, bool use_bn> struct feed_forwards_impl
	{
	public:
		feed_forwards_impl()
		{
			using layer = get_layer<l>;
			if (use_dropout && l != 0 && layer::type != CNN_LAYER_SOFTMAX)
				dropout<l>();
			layer::feed_forwards(get_layer<l + 1>::feature_maps);
			if (use_bn)
			{
				for (size_t f = 0; f < layer::activations_mean.size(); ++f)
				{
					for (size_t i = 0; i < layer::activations_mean.rows(); ++i)
					{
						for (size_t j = 0; j < layer::activations_mean.cols(); ++j)
						{
							float old_mean = layer::activations_mean[f].at(i, j);
							float old_var = layer::activations_variance[f].at(i, j);
							float x = layer::feature_maps[f].at(i, j);
							float new_mean = old_mean * (n_bn - 1) / n_bn + x / n_bn;
							layer::activations_mean[f].at(i, j) = new_mean;
							layer::activations_variance[f].at(i, j) = (x * x + (n_bn - 1) * (old_var + old_mean * old_mean)) / n_bn - new_mean * new_mean;
						}
					}
				}
			}
		}
	};

	template<size_t l, bool sample> struct feed_backwards_impl
	{
	public:
		feed_backwards_impl()
		{
			using layer = get_layer<l>;
			layer::feed_backwards(get_layer<l + 1>::feature_maps);
			if (sample)
				layer::stochastic_sample(layer::feature_maps);
		}
	};

	template<size_t l> struct apply_batch_normalization_impl
	{
	public:
		apply_batch_normalization_impl()
		{
			using t = decltype(get_layer<l>::feature_maps);
			for (size_t f = 0; f < t::size(); ++f)
				for (size_t i = 0; i < t::rows(); ++i)
					for (size_t j = 0; j < t::cols(); ++j)
						get_layer<l>::feature_maps[f].at(i, j) = (get_layer<l>::feature_maps[f].at(i, j) - get_layer<l>::activations_mean[f].at(i, j)) / (float)sqrt(1e-8f + get_layer<l>::activations_variance[f].at(i, j)); //TODO: add gamma and beta and temp
		}
	};

	template<size_t l> struct backprop_impl
	{
	public:
		backprop_impl()
		{
			get_layer<l>::back_prop(get_layer<l - 1>::activation, get_layer<l + 1>::feature_maps,
				!use_batch_learning && optimization_method == CNN_OPT_BACKPROP, learning_rate,
				use_hessian, minimum_divisor, use_momentum && !use_batch_learning, momentum_term,
				use_l2_weight_decay, include_bias_decay, weight_decay_factor);
		}
	};

	template<size_t l> struct add_weight_decay_impl
	{
	public:
		add_weight_decay_impl()
		{
			if (!include_bias_decay)
			{
				using t = decltype(get_layer<l>::weights_gradient);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							get_layer<l>::weights_gradient[d].at(i, j) += 2 * weight_decay_factor * get_layer<l>::weights[d].at(i, j);
			}

			else
			{
				using t = decltype(get_layer<l>::biases_gradient);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							get_layer<l>::biases_gradient[f_0].at(i_0, j_0) += 2 * weight_decay_factor * get_layer<l>::biases[f_0].at(i_0, j_0);
			}
		}
	};

	template<size_t l> struct apply_grad_impl
	{
	public:
		apply_grad_impl()
		{
			using layer = get_layer<l>;
			using weights_t = decltype(layer::weights);
			using biases_t = decltype(layer::biases);
			if (use_hessian)
			{
				for (size_t d = 0; d < weights_t::size(); ++d)
					for (size_t i = 0; i < weights_t::rows(); ++i)
						for (size_t j = 0; j < weights_t::cols(); ++j)
							layer::weights_gradient[d].at(i, j) /= layer::weights_hessian[d].at(i, j);
				for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) /= layer::biases_hessian[f_0].at(i_0, j_0);
			}

			if (use_momentum)
			{
				//update weights
				for (size_t d = 0; d < weights_t::size(); ++d)
				{
					for (size_t i = 0; i < weights_t::rows(); ++i)
					{
						for (size_t j = 0; j < weights_t::cols(); ++j)
						{
							layer::weights[d].at(i, j) += -learning_rate * layer::weights_gradient[d].at(i, j) + momentum_term * layer::weights_momentum[d].at(i, j);
							layer::weights_momentum[d].at(i, j) = momentum_term * layer::weights_momentum[d].at(i, j) + -learning_rate * layer::weights_gradient[d].at(i, j);
							layer::weights_gradient[d].at(i, j) = 0;
						}
					}
				}

				//update biases
				for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
				{
					for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
					{
						for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
						{
							layer::biases[f_0].at(i_0, j_0) += -learning_rate * layer::biases_gradient[f_0].at(i_0, j_0) + momentum_term * layer::biases_momentum[f_0].at(i_0, j_0);
							layer::biases_momentum[f_0].at(i_0, j_0) = momentum_term * layer::biases_momentum[f_0].at(i_0, j_0) + -learning_rate * layer::biases_gradient[f_0].at(i_0, j_0);
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}

			else if (optimization_method == CNN_OPT_ADAM)
			{
				//update weights
				for (size_t d = 0; d < weights_t::size(); ++d)
				{
					for (size_t i = 0; i < weights_t::rows(); ++i)
					{
						for (size_t j = 0; j < weights_t::cols(); ++j)
						{
							float g = layer::weights_gradient[d].at(i, j);
							layer::weights_momentum[d].at(i, j) = beta1 * layer::weights_momentum[d].at(i, j) + (1.0f - beta1) * g;
							layer::weights_hessian[d].at(i, j) = beta2 * layer::weights_hessian[d].at(i, j) + (1.0f - beta2) * g * g;
							layer::weights[d].at(i, j) += -learning_rate * (float)sqrt(1.0f - pow(beta2, t_adam)) / (1.0f - (float)pow(beta1, t_adam)) * layer::weights_momentum[d].at(i, j) / ((float)sqrt(layer::weights_hessian[d].at(i, j)) + 1e-7f);
							layer::weights_gradient[d].at(i, j) = 0;
						}
					}
				}

				//update biases
				for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
				{
					for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
					{
						for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
						{
							float g = layer::biases_gradient[f_0].at(i_0, j_0);
							layer::biases_momentum[f_0].at(i_0, j_0) = beta1 * layer::biases_momentum[f_0].at(i_0, j_0) + (1 - beta1) * g;
							layer::biases_hessian[f_0].at(i_0, j_0) = beta2 * layer::biases_hessian[f_0].at(i_0, j_0) + (1 - beta2) * g * g;
							layer::biases[f_0].at(i_0, j_0) += -learning_rate * (float)sqrt(1 - pow(beta2, t_adam)) / (float)(1 - pow(beta1, t_adam)) * layer::biases_momentum[f_0].at(i_0, j_0) / (float)(sqrt(layer::biases_hessian[f_0].at(i_0, j_0)) + 1e-7f);
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}

			else if (optimization_method == CNN_OPT_ADAGRAD)
			{
				//update weights
				for (size_t d = 0; d < weights_t::size(); ++d)
				{
					for (size_t i = 0; i < weights_t::rows(); ++i)
					{
						for (size_t j = 0; j < weights_t::cols(); ++j)
						{
							float g = layer::weights_gradient[d].at(i, j);
							layer::weights[d].at(i, j) += -learning_rate / sqrt(layer::weights_hessian[d].at(i, j) + minimum_divisor) * g;
							layer::weights_hessian[d].at(i, j) += g * g;
							layer::weights_gradient[d].at(i, j) = 0;
						}
					}
				}

				//update biases
				for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
				{
					for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
					{
						for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
						{
							float g = layer::biases_gradient[f_0].at(i_0, j_0);
							layer::biases[f_0].at(i_0, j_0) += -learning_rate / sqrt(layer::biases_hessian[f_0].at(i_0, j_0) + minimum_divisor) * g;
							layer::biases_hessian[f_0].at(i_0, j_0) += g * g;
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}

			else
			{
				//update weights
				for (size_t d = 0; d < weights_t::size(); ++d)
				{
					for (size_t i = 0; i < weights_t::rows(); ++i)
					{
						for (size_t j = 0; j < weights_t::cols(); ++j)
						{
							layer::weights[d].at(i, j) += -learning_rate * layer::weights_gradient[d].at(i, j);
							layer::weights_gradient[d].at(i, j) = 0;
						}
					}
				}

				//update biases
				for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
				{
					for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
					{
						for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
						{
							layer::biases[f_0].at(i_0, j_0) += -learning_rate * layer::biases_gradient[f_0].at(i_0, j_0);
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}
		}
	};

public:

	//Hyperparameters

	static constexpr size_t num_layers = sizeof...(layers);
	static constexpr size_t last_layer_index = num_layers - 1;

	//incremental loop
	template<template<size_t> class loop_body> using loop_up_layers = for_loop<0, last_layer_index - 1, 1, loop_body>;
	//decremental loop
	template<template<size_t> class loop_body> using loop_down_layers = for_loop<last_layer_index, 1, 1, loop_body>;
	//fetch a layer with a constexpr
	template<size_t l> using get_layer = get_type<l, layers...>;

	//learning rate (should be positive)
	static float learning_rate;
	//only set if using hessian
	static float minimum_divisor;
	//only set if using momentum 
	static float momentum_term;
	//only set if using dropout. This proportion of neurons will be "dropped"
	static float dropout_probability;
	//must be set if using Adam
	static float beta1;
	//must be set if using Adam
	static float beta2;
	//must be set if using L2 weight decay
	static float weight_decay_factor;

	static FeatureMaps<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> input;
	static FeatureMaps<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()> labels;

	//Loop bodies

	template<typename file> using save_net_data = save_data_t<file>;
	template<typename file> using load_net_data = load_data_t<file>;

	template<size_t l> using reset_layer_feature_maps = reset_impl<l, CNN_DATA_FEATURE_MAP>;
	template<size_t l> using reset_layer_weights_gradient = reset_impl<l, CNN_DATA_WEIGHT_GRAD>;
	template<size_t l> using reset_layer_biases_gradient = reset_impl<l, CNN_DATA_BIAS_GRAD>;
	template<size_t l> using reset_layer_weights_momentum = reset_impl<l, CNN_DATA_WEIGHT_MOMENT>;
	template<size_t l> using reset_layer_biases_momentum = reset_impl<l, CNN_DATA_BIAS_MOMENT>;
	template<size_t l> using reset_layer_weights_hessian = reset_impl<l, CNN_DATA_WEIGHT_HESSIAN>;
	template<size_t l> using reset_layer_biases_hessian = reset_impl<l, CNN_DATA_BIAS_HESSIAN>;
	template<size_t l> using reset_layer_activations = reset_impl<l, CNN_DATA_ACT>;

	template<size_t l> using delete_layer_feature_maps = delete_impl<l, CNN_DATA_FEATURE_MAP>;
	template<size_t l> using delete_layer_weights_momentum = delete_impl<l, CNN_DATA_WEIGHT_MOMENT>;
	template<size_t l> using delete_layer_biases_momentum = delete_impl<l, CNN_DATA_BIAS_MOMENT>;
	template<size_t l> using delete_layer_weights_hessian = delete_impl<l, CNN_DATA_WEIGHT_HESSIAN>;
	template<size_t l> using delete_layer_biases_hessian = delete_impl<l, CNN_DATA_BIAS_HESSIAN>;
	template<size_t l> using delete_layer_activations = delete_impl<l, CNN_DATA_ACT>;

	template<size_t l> using feed_forwards_layer = feed_forwards_impl<l, use_batch_normalization && !collect_data_while_training && keep_running_activation_statistics>;
	template<size_t l> using feed_forwards_training_layer = feed_forwards_impl<l, use_batch_normalization && collect_data_while_training && !keep_running_activation_statistics>;

	template<size_t l> using feed_backwards_layer_nosample = feed_backwards_impl<l, false>;
	template<size_t l> using feed_backwards_layer_sample = feed_backwards_impl<l, true>;

	template<size_t l> using apply_batch_normalization_layer = apply_batch_normalization_impl<l>;

	template<size_t l> using back_prop_layer = backprop_impl<l>;

	template<size_t l> using add_weight_decay_layer = add_weight_decay_impl<l>;

	template<size_t l> using apply_gradient_layer = apply_grad_impl<l>;

private:
	//used for adam
	static size_t t_adam;
	//used for batch normalization
	static size_t n_bn;

	static constexpr size_t last_rbm_index = get_rbm_idx<layers...>::idx;

	template<size_t l> static void dropout();

	static FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> error_signals();

	//TODO: implement or get rid of
	static void hessian_error_signals();

public:

	//Functions

	NeuralNet() = default;

	~NeuralNet() = default;

	//setup net, must do
	static void setup();

	//save learned net
	template<typename file_name_type> static void save_data();

	//load previously learned net
	template<typename file_name_type> static void load_data();

	//set input (for discrimination)
	static void set_input(FeatureMaps<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& new_input);

	//set labels for batch
	static void set_labels(FeatureMaps<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& new_labels);

	//feed forwards
	static void discriminate();

	//feed backwards, returns a copy of the first layer (must be deallocated)
	static FeatureMaps<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> generate(FeatureMaps<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& input, size_t iterations, bool use_sampling);

	//wake-sleep algorithm, only trains target layer with assumption that layers up to it have been trained
	static void pretrain(size_t markov_iterations);

	//backpropogate with selected method, returns error by loss function
	static float train();

	//backprop for a batch with selected method, returns mean error by loss function
	static float train_batch(std::vector<FeatureMaps<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>>& batch_input, std::vector<FeatureMaps<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>>& batch_labels);

	//update second derivatives TODO: implement or get rid of
	static void calculate_hessian(bool use_first_deriv, float gamma);

	//reset and apply gradient																  
	static void apply_gradient();

	//get current error according to loss function
	static float global_error();
};

//Hyperparameter declarations

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::learning_rate = .001f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::minimum_divisor = .1f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::momentum_term = .8f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::dropout_probability = .5f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::beta1 = .9f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::beta2 = .99f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::weight_decay_factor = .001f;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> size_t NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::t_adam = 0;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> size_t NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::n_bn = 0;
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> template<typename file_name_type> FILE* NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::save_data_t<file_name_type>::fp = {};
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> template<typename file_name_type> FILE* NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::load_data_t<file_name_type>::fp = {};
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> FeatureMaps<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()> NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::input = {};
template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers> FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::labels = {};

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
setup()
{
	//conditions
	if (optimization_method == CNN_OPT_ADAM)
	{
		static_assert(use_momentum == false, "error, need to change optimization method or use_momentum");
		static_assert(use_hessian == false, "error, need to change optimization method or use_hessian");
	}

	if (optimization_method == CNN_OPT_ADAGRAD)
	{
		static_assert(use_hessian == false, "error, need to change optimization method or use_hessian");
	}

	//Don't need this since there is implicit initialization
	/*if (!use_momentum && optimization_method != CNN_OPT_ADAM)
	{
	loop_up_layers<delete_layer_weights_momentum>();
	loop_up_layers<delete_layer_biases_momentum>();
	}

	if (!use_hessian && optimization_method == CNN_OPT_BACKPROP)
	{
	loop_up_layers<delete_layer_weights_hessian>();
	loop_up_layers<delete_layer_biases_hessian>();
	}

	if (!use_batch_normalization)
	{
	loop_up_layers<delete_layer_activations>();
	}*/
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
template<typename file_name_type>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
save_data()
{
	save_net_data<file_name_type>();
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
template<typename file_name_type>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
load_data()
{
	load_net_data<file_name_type>();
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
set_input(FeatureMaps<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()>& new_input)
{
	for (size_t f = 0; f < input.size(); ++f)
	{
		for (size_t i = 0; i < input.rows(); ++i)
		{
			for (size_t j = 0; j < input.cols(); ++j)
			{
				input[f].at(i, j) = new_input[f].at(i, j);
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
			}
		}
	}
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
set_labels(FeatureMaps<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& new_labels)
{
	for (size_t f = 0; f < labels.size(); ++f)
		for (size_t i = 0; i < labels.rows(); ++i)
			for (size_t j = 0; j < labels.cols(); ++j)
				labels[f].at(i, j) = new_labels[f].at(i, j);
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
discriminate()
{
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	for_loop<1, last_layer_index, 1, reset_layer_feature_maps>();
	loop_up_layers<feed_forwards_layer>();
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline FeatureMaps<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()> NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
generate(FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& input, size_t iterations, bool use_sampling)
{
	//reset all but output (or inputs?)
	loop_up_layers<reset_layer_feature_maps>();
	get_layer<last_layer_index>::feed_backwards(input);

	for_loop<last_layer_index - 1, last_rbm_index, 1, feed_backwards_layer_nosample>();
	using rbm_layer = get_layer<last_rbm_index>;

	//gibbs sample
	rbm_layer::feed_backwards(get_layer<last_rbm_index + 1>::feature_maps);
	for (size_t i = 0; i < iterations; ++i)
	{
		if (use_sampling)
			rbm_layer::stochastic_sample(rbm_layer::feature_maps);
		rbm_layer::feed_forwards(get_layer<last_rbm_index + 1>::feature_maps);
		get_layer<last_rbm_index + 1>::stochastic_sample(get_layer<last_rbm_index + 1>::feature_maps);
		rbm_layer::feed_backwards(get_layer<last_rbm_index + 1>::feature_maps);
	}

	if (use_sampling)
		for_loop<last_rbm_index - 1, 0, 1, feed_backwards_layer_sample>();
	else
		for_loop<last_rbm_index - 1, 0, 1, feed_backwards_layer_nosample>();
	FeatureMaps<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> output = {};
	for (size_t f = 0; f < output.size(); ++f)
		output[f] = get_layer<0>::feature_maps[f].clone();
	return output;
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
pretrain(size_t markov_iterations)
{
	//reset input
	loop_up_layers<reset_layer_feature_maps>();
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	loop_up_layers<feed_forwards_training_layer>();
	if (use_batch_normalization && collect_data_while_training && !keep_running_activation_statistics)
		++n_bn;

	using target_layer = get_layer<last_layer_index>; //todo add in target layer
	if (target_layer::type == CNN_LAYER_CONVOLUTION || target_layer::type == CNN_LAYER_PERCEPTRONFULLCONNECTIVITY)
		target_layer::wake_sleep(learning_rate, use_dropout, markov_iterations);
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
train()
{
	float error;
	loop_up_layers<reset_layer_feature_maps>();
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	loop_up_layers<feed_forwards_training_layer>();
	if (use_batch_normalization && collect_data_while_training && !keep_running_activation_statistics)
		++n_bn;

	error = global_error();

	//values of the network when fed forward
	if (use_batch_normalization)
		loop_up_layers<apply_batch_normalization_layer>();

	//get error signals for output and returns any layers to be skipped
	constexpr size_t off = (loss_function == CNN_LOSS_LOGLIKELIHOOD ? 1 : 0);
	auto errors = error_signals();

	//backprop for each layer (need to get activation derivatives for output first
	get_layer<last_layer_index>::back_prop(get_layer<last_layer_index - 1>::activation, errors,
		!use_batch_learning && optimization_method == CNN_OPT_BACKPROP, learning_rate,
		use_hessian, minimum_divisor, use_momentum && !use_batch_learning, momentum_term,
		use_l2_weight_decay, include_bias_decay, weight_decay_factor);
	for_loop<last_layer_index - 1 - off, 1, 1, back_prop_layer>();

	if (!use_batch_learning && optimization_method != CNN_OPT_BACKPROP)
		apply_gradient();

	return error;
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
train_batch(std::vector<FeatureMaps<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()>>& batch_inputs, std::vector<FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>>& batch_labels)
{
	bool temp_batch = use_batch_learning;
	use_batch_learning = true;
	float total_error = 0.0f;
	for (size_t i = 0; i < batch_inputs.size(); ++i)
	{
		set_input(batch_inputs[i]);
		set_labels(batch_labels[i]);
		total_error += train();
	}
	apply_gradient();
	use_batch_learning = temp_batch;
	return total_error / batch_inputs.size();
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
apply_gradient()
{
	if (use_l2_weight_decay && use_batch_learning)
		loop_up_layers<add_weight_decay_layer>();

	if (optimization_method == CNN_OPT_ADAM)
		++t_adam;
	if (use_batch_normalization && !keep_running_activation_statistics)
	{
		n_bn = 0;
		loop_up_layers<reset_layer_activations>();
	}
	loop_up_layers<apply_gradient_layer>();
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline float NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
global_error()
{
	float sum = 0.0f;

	if (loss_function == CNN_LOSS_SQUAREERROR)
	{
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels[f].rows(); ++i)
				for (size_t j = 0; j < labels[f].cols(); ++j)
					sum += pow(labels[f].at(i, j) - get_layer<last_layer_index>::feature_maps[f].at(i, j), 2);
		return sum / 2;
	}
	else if (loss_function == CNN_LOSS_LOGLIKELIHOOD)
	{
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels[f].rows(); ++i)
				for (size_t j = 0; j < labels[f].cols(); ++j)
					if (labels[f].at(i, j) > 0)
						return -log(get_layer<last_layer_index>::feature_maps[f].at(i, j));
	}
	else if (loss_function == CNN_LOSS_CUSTOMTARGETS)
		return 0;
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
template<size_t l>
inline void NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
dropout()
{
	using layer = get_layer<l>;
	for (size_t f = 0; f < layer::feature_maps.size(); ++f)
		for (size_t i = 0; i < layer::feature_maps.rows(); ++i)
			for (size_t j = 0; j < layer::feature_maps.cols(); ++j)
				if ((1.0f * rand()) / RAND_MAX <= dropout_probability)
					layer::feature_maps[f].at(i, j) = 0;
}

template<size_t loss_function, size_t optimization_method, bool use_dropout, bool use_batch_learning, bool use_momentum, bool use_hessian, bool use_l2_weight_decay, bool include_bias_decay, bool use_batch_normalization, bool keep_running_activation_statistics, bool collect_data_while_training, typename... layers>
inline FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> NeuralNet<loss_function, optimization_method, use_dropout, use_batch_learning, use_momentum, use_hessian, use_l2_weight_decay, include_bias_decay, use_batch_normalization, keep_running_activation_statistics, collect_data_while_training, layers...>::
error_signals()
{
	auto out = FeatureMaps<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>{ 0 };
	if (loss_function == CNN_LOSS_SQUAREERROR)
		for (size_t k = 0; k < labels.size(); ++k)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[k].at(i, j) = get_layer<last_layer_index>::feature_maps[k].at(i, j) - labels[k].at(i, j);
	else if (loss_function == CNN_LOSS_LOGLIKELIHOOD) //assumes next layer is softmax
	{
		for (size_t k = 0; k < labels.size(); ++k)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[k].at(i, j) = get_layer<last_layer_index - 1>::feature_maps[k].at(i, j) - labels[k].at(i, j);
	}
	else if (loss_function == CNN_LOSS_CUSTOMTARGETS)
		for (size_t k = 0; k < labels.size(); ++k)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[k].at(i, j) = labels[k].at(i, j);
	return out;
}