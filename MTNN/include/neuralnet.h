#pragma once

#include <cstring>
#include <functional>
#include <stdio.h>
#include <tuple>

#include "imatrix.h"
#include "ilayer.h"

//default, MSE
#define MTNN_LOSS_SQUAREERROR 0
//assumes prior layer is softmax
#define MTNN_LOSS_LOGLIKELIHOOD 1
//undefined for error, instead sets output to labels during training
#define MTNN_LOSS_CUSTOMTARGETS 2

//vanilla, add in momentum or hessian if desired
#define MTNN_OPT_BACKPROP 0
//can't use with momentum or hessian
#define MTNN_OPT_ADAM 1
//can't use with momentum or hessian
#define MTNN_OPT_ADAGRAD 2

//HELPER FUNCTIONS //// Network class definitions begin at line 147

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
	static constexpr size_t idx = (get_type<n, Ts...>::activation == MTNN_FUNC_RBM) ? n : get_rbm_idx_impl<n - 1, Ts...>::idx;
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

template<typename... layers>
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
			void write_float(const float& f, FILE* file)
			{
				fwrite(&f, sizeof(float), 1, file);
			}
		public:
			save_data_impl()
			{
				using layer = get_layer<l>;

				if (layer::type == MTNN_LAYER_BATCHNORMALIZATION)
				{
					using t = decltype(layer::activations_population_mean);
					for (size_t d = 0; d < t::size(); ++d)
					{
						for (size_t i = 0; i < t::rows(); ++i)
						{
							for (size_t j = 0; j < t::cols(); ++j)
							{
								write_float(layer::activations_population_mean[d].at(i, j), fp);
								write_float(layer::activations_population_variance[d].at(i, j), fp);
							}
						}
					}
				}

				//begin weights values
				{
					using t = decltype(layer::weights);
					for (size_t d = 0; d < t::size(); ++d)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								write_float(layer::weights[d].at(i, j), fp);
				}

				//begin biases values
				{
					using t = decltype(layer::biases);
					for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
						for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
							for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
								write_float(layer::biases[f_0].at(i_0, j_0), fp);//bias values
				}

				//begin gen biases values
				{
					using t = decltype(layer::generative_biases);
					for (size_t f = 0; f < t::size(); ++f)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								write_float(layer::generative_biases[f].at(i, j), fp);//gen bias values
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
			void read_float(float& out_float, FILE* file)
			{
				fread(&out_float, sizeof(float), 1, file);
			}
		public:
			load_data_impl()
			{
				using layer = get_layer<l>;

				if (layer::type == MTNN_LAYER_BATCHNORMALIZATION)
				{
					using t = decltype(layer::activations_population_mean);
					for (size_t d = 0; d < t::size(); ++d)
					{
						for (size_t i = 0; i < t::rows(); ++i)
						{
							for (size_t j = 0; j < t::cols(); ++j)
							{
								read_float(layer::activations_population_mean[d].at(i, j), fp);
								read_float(layer::activations_population_variance[d].at(i, j), fp);
							}
						}
					}
				}

				//begin weights values
				{
					using t = decltype(layer::weights);
					for (size_t d = 0; d < t::size(); ++d)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								read_float(layer::weights[d].at(i, j), fp);
				}

				//begin biases values
				{
					using t = decltype(layer::biases);
					for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
						for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
							for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
								read_float(layer::biases[f_0].at(i_0, j_0), fp);
				}

				//begin gen biases values
				{
					using t = decltype(layer::generative_biases);
					for (size_t f = 0; f < t::size(); ++f)
						for (size_t i = 0; i < t::rows(); ++i)
							for (size_t j = 0; j < t::cols(); ++j)
								read_float(layer::generative_biases[f].at(i, j), fp);
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
			if (target == MTNN_DATA_FEATURE_MAP)
			{
				using t = decltype(layer::feature_maps);
				for (size_t f = 0; f < t::size(); ++f)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::feature_maps[f].at(i, j) = 0.0f;
				//reset batch data
				for (size_t in = 0; in < get_batch_activations<l>().size(); ++in)
				{
					for (size_t f = 0; f < t::size(); ++f)
					{
						for (size_t i = 0; i < t::rows(); ++i)
						{
							for (size_t j = 0; j < t::cols(); ++j)
							{
								get_batch_activations<l>()[in][f].at(i, j) = 0;
								get_batch_out_derivs<l>()[in][f].at(i, j) = 0;
							}
						}
					}
				}
			}
			if (target == MTNN_DATA_WEIGHT_GRAD)
			{
				using t = decltype(layer::weights_gradient);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_gradient[d].at(i, j) = 0.0f;
			}
			if (target == MTNN_DATA_BIAS_GRAD)
			{
				using t = decltype(layer::biases_gradient);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) = 0.0f;
			}
			if (target == MTNN_DATA_WEIGHT_MOMENT)
			{
				using t = decltype(layer::weights_momentum);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_momentum[d].at(i, j) = 0.0f;
			}
			if (target == MTNN_DATA_BIAS_MOMENT)
			{
				using t = decltype(layer::biases_gradient);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) = 0.0f;
			}
			if (target == MTNN_DATA_WEIGHT_AUXDATA)
			{
				using t = decltype(layer::weights_aux_data);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_aux_data[d].at(i, j) = 0.0f;
			}
			if (target == MTNN_DATA_BIAS_AUXDATA)
			{
				using t = decltype(layer::biases_aux_data);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_aux_data[f_0].at(i_0, j_0) = 0.0f;
			}
		}
	};

	template<size_t l, size_t target> struct delete_impl
	{
	public:
		delete_impl()
		{
			using layer = get_layer<l>;
			if (target == MTNN_DATA_FEATURE_MAP)
			{
				using t = decltype(layer::feature_maps);
				layer::feature_maps.~FeatureMap<t::size(), t::rows(), t::cols()>();
			}
			if (target == MTNN_DATA_WEIGHT_MOMENT)
			{
				using t = decltype(layer::weights_momentum);
				layer::weights_momentum.~FeatureMap<t::size(), t::rows(), t::cols()>();
			}
			if (target == MTNN_DATA_BIAS_MOMENT)
			{
				using t = decltype(layer::biases_momentum);
				layer::biases_momentum.~FeatureMap<t::size(), t::rows(), t::cols()>();
			}
			if (target == MTNN_DATA_WEIGHT_AUXDATA)
			{
				using t = decltype(layer::weights_aux_data);
				layer::weights_aux_data.~FeatureMap<t::size(), t::rows(), t::cols()>();
			}
			if (target == MTNN_DATA_BIAS_AUXDATA)
			{
				using t = decltype(layer::biases_aux_data);
				layer::biases_aux_data.~FeatureMap<t::size(), t::rows(), t::cols()>();
			}
		}
	};

	template<size_t l, bool training> struct feed_forwards_impl
	{
	public:
		feed_forwards_impl()
		{
			using layer = get_layer<l>;
			if (use_dropout && l != 0 && layer::type != MTNN_LAYER_SOFTMAX)
				dropout<l>();
			layer::feed_forwards(layer::feature_maps, get_layer<l + 1>::feature_maps);

			//get activations
			auto& src = layer::feature_maps;
			auto& dest = get_batch_activations<l>()[0];
			for (size_t f = 0; f < src.size(); ++f)
			{
				for (size_t i = 0; i < src.rows(); ++i)
				{
					for (size_t j = 0; j < src.cols(); ++j)
					{
						dest[f].at(i, j) = src[f].at(i, j);
						src[f].at(i, j) = 0;//reset for next iteration
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

	template<size_t l> struct back_prop_impl
	{
	public:
		back_prop_impl()
		{
			get_layer<l>::back_prop(get_layer<l - 1>::activation, get_layer<l + 1>::feature_maps,
				get_batch_activations<l>()[0], get_layer<l>::feature_maps,
				!use_batch_learning && optimization_method == MTNN_OPT_BACKPROP, learning_rate,
				use_momentum && !use_batch_learning, momentum_term,
				use_l2_weight_decay, include_bias_decay, weight_decay_factor);
		}
	};

	template<size_t l, bool training> struct feed_forwards_batch_impl
	{
	public:
		feed_forwards_batch_impl()
		{
			if (use_dropout && l != 0 && get_layer<l>::type != MTNN_LAYER_SOFTMAX)
				dropout<l>();//todo vec also training bool
			get_layer<l>::feed_forwards(get_batch_activations<l>(), get_batch_activations<l + 1>());
		}
	};

	template<size_t l, bool sample> struct feed_backwards_batch_impl
	{
	public:
		feed_backwards_batch_impl()
		{
			get_layer<l>::feed_backwards(get_batch_activations<l + 1>(), get_batch_activations<l>());
			if (sample)
				layer::stochastic_sample(layer::feature_maps);//todo vec
		}
	};

	template<size_t l> struct back_prop_batch_impl
	{
	public:
		back_prop_batch_impl()
		{
			get_layer<l>::back_prop(get_layer<l - 1>::activation, get_batch_out_derivs<l + 1>(),
				get_batch_activations<l>(), get_batch_out_derivs<l>(),
				!use_batch_learning && optimization_method == MTNN_OPT_BACKPROP, learning_rate,
				use_momentum && !use_batch_learning, momentum_term,
				use_l2_weight_decay, include_bias_decay, weight_decay_factor);
		}
	};

	template<size_t l> struct feed_forwards_pop_stats_impl
	{
	public:
		feed_forwards_pop_stats_impl()
		{
			using layer = get_layer<l>;

			//calculate statistics for batch normalization layer
			auto& inputs = get_batch_activations<l>();
			auto& outputs = get_batch_activations<l + 1>();
			if (layer::type == MTNN_LAYER_BATCHNORMALIZATION)
			{
				using t = decltype(layer::feature_maps);
				for (size_t f = 0; f < t::size(); ++f)
				{
					for (size_t i = 0; i < t::rows(); ++i)
					{
						for (size_t j = 0; j < t::cols(); ++j)
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

							//store stats
							float mean = sumx / n_in;
							layer::activations_population_mean[f].at(i, j) = mean;
							layer::activations_population_variance[f].at(i, j) = sumxsqr / n_in - mean * mean;
						}
					}
				}
			}

			//can't feed forward batch because batch norm will use sample statistics
			for (size_t in = 0; in < inputs.size(); ++in)
				layer::feed_forwards(inputs[in], outputs[in]);
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
			/*if (use_hessian)
			{
			for (size_t d = 0; d < weights_t::size(); ++d)
			for (size_t i = 0; i < weights_t::rows(); ++i)
			for (size_t j = 0; j < weights_t::cols(); ++j)
			layer::weights_gradient[d].at(i, j) /= layer::weights_aux_data[d].at(i, j);
			for (size_t f_0 = 0; f_0 < biases_t::size(); ++f_0)
			for (size_t i_0 = 0; i_0 < biases_t::rows(); ++i_0)
			for (size_t j_0 = 0; j_0 < biases_t::cols(); ++j_0)
			layer::biases_gradient[f_0].at(i_0, j_0) /= layer::biases_aux_data[f_0].at(i_0, j_0);
			}*/

			if (optimization_method == MTNN_OPT_ADAM && layer::type != MTNN_LAYER_BATCHNORMALIZATION)
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
							layer::weights_aux_data[d].at(i, j) = beta2 * layer::weights_aux_data[d].at(i, j) + (1.0f - beta2) * g * g;
							layer::weights[d].at(i, j) += -learning_rate * (float)sqrt(1.0f - pow(beta2, t_adam)) / (1.0f - (float)pow(beta1, t_adam)) * layer::weights_momentum[d].at(i, j) / ((float)sqrt(layer::weights_aux_data[d].at(i, j)) + 1e-7f);
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
							layer::biases_aux_data[f_0].at(i_0, j_0) = beta2 * layer::biases_aux_data[f_0].at(i_0, j_0) + (1 - beta2) * g * g;
							layer::biases[f_0].at(i_0, j_0) += -learning_rate * (float)sqrt(1 - pow(beta2, t_adam)) / (float)(1 - pow(beta1, t_adam)) * layer::biases_momentum[f_0].at(i_0, j_0) / (float)(sqrt(layer::biases_aux_data[f_0].at(i_0, j_0)) + 1e-7f);
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}

			else if (optimization_method == MTNN_OPT_ADAGRAD && layer::type != MTNN_LAYER_BATCHNORMALIZATION)
			{
				//update weights
				for (size_t d = 0; d < weights_t::size(); ++d)
				{
					for (size_t i = 0; i < weights_t::rows(); ++i)
					{
						for (size_t j = 0; j < weights_t::cols(); ++j)
						{
							float g = layer::weights_gradient[d].at(i, j);
							layer::weights[d].at(i, j) += -learning_rate / sqrt(layer::weights_aux_data[d].at(i, j) + minimum_divisor) * g;
							layer::weights_aux_data[d].at(i, j) += g * g;
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
							layer::biases[f_0].at(i_0, j_0) += -learning_rate / sqrt(layer::biases_aux_data[f_0].at(i_0, j_0) + minimum_divisor) * g;
							layer::biases_aux_data[f_0].at(i_0, j_0) += g * g;
							layer::biases_gradient[f_0].at(i_0, j_0) = 0;
						}
					}
				}
			}

			else if (use_momentum)
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

	template<size_t l, bool add> struct modify_batch_activations_vector_impl
	{
		modify_batch_activations_vector_impl()
		{
			if (add)
				get_batch_activations<l>().push_back(std::remove_reference_t<decltype(get_batch_activations<l>())>::value_type{ 0 });
			else
				get_batch_activations<l>().pop_back();
		}
	};

	template<size_t l, bool add> struct modify_batch_out_derivs_vector_impl
	{
		modify_batch_out_derivs_vector_impl()
		{
			if (add)
				get_batch_out_derivs<l>().push_back(std::remove_reference_t<decltype(get_batch_out_derivs<l>())>::value_type{ 0 });
			else
				get_batch_out_derivs<l>().pop_back();
		}
	};

	//Loop bodies

	template<typename file> using save_net_data = save_data_t<file>;
	template<typename file> using load_net_data = load_data_t<file>;

	template<size_t l> using reset_layer_feature_maps = reset_impl<l, MTNN_DATA_FEATURE_MAP>;
	template<size_t l> using reset_layer_weights_gradient = reset_impl<l, MTNN_DATA_WEIGHT_GRAD>;
	template<size_t l> using reset_layer_biases_gradient = reset_impl<l, MTNN_DATA_BIAS_GRAD>;
	template<size_t l> using reset_layer_weights_momentum = reset_impl<l, MTNN_DATA_WEIGHT_MOMENT>;
	template<size_t l> using reset_layer_biases_momentum = reset_impl<l, MTNN_DATA_BIAS_MOMENT>;
	template<size_t l> using reset_layer_weights_aux_data = reset_impl<l, MTNN_DATA_WEIGHT_AUXDATA>;
	template<size_t l> using reset_layer_biases_aux_data = reset_impl<l, MTNN_DATA_BIAS_AUXDATA>;

	template<size_t l> using delete_layer_feature_maps = delete_impl<l, MTNN_DATA_FEATURE_MAP>;
	template<size_t l> using delete_layer_weights_momentum = delete_impl<l, MTNN_DATA_WEIGHT_MOMENT>;
	template<size_t l> using delete_layer_biases_momentum = delete_impl<l, MTNN_DATA_BIAS_MOMENT>;
	template<size_t l> using delete_layer_weights_aux_data = delete_impl<l, MTNN_DATA_WEIGHT_AUXDATA>;
	template<size_t l> using delete_layer_biases_aux_data = delete_impl<l, MTNN_DATA_BIAS_AUXDATA>;

	template<size_t l> using feed_forwards_layer = feed_forwards_impl<l, false>;
	template<size_t l> using feed_forwards_training_layer = feed_forwards_impl<l, true>;

	template<size_t l> using feed_forwards_batch_layer = feed_forwards_batch_impl<l, false>;
	template<size_t l> using feed_forwards_batch_training_layer = feed_forwards_batch_impl<l, true>;

	template<size_t l> using feed_forwards_population_statistics_layer = feed_forwards_pop_stats_impl<l>;

	template<size_t l> using feed_backwards_layer_nosample = feed_backwards_impl<l, false>;
	template<size_t l> using feed_backwards_layer_sample = feed_backwards_impl<l, true>;

	template<size_t l> using feed_backwards_batch_layer_nosample = feed_backwards_batch_impl<l, false>;
	template<size_t l> using feed_backwards_batch_layer_sample = feed_backwards_batch_impl<l, true>;

	template<size_t l> using back_prop_layer = back_prop_impl<l>;

	template<size_t l> using back_prop_batch_layer = back_prop_batch_impl<l>;

	template<size_t l> using add_weight_decay_layer = add_weight_decay_impl<l>;

	template<size_t l> using apply_gradient_layer = apply_grad_impl<l>;

	template<size_t l> using add_batch_activations = modify_batch_activations_vector_impl<l, true>;
	template<size_t l> using remove_batch_activations = modify_batch_activations_vector_impl<l, false>;

	template<size_t l> using add_batch_out_derivs = modify_batch_out_derivs_vector_impl<l, true>;
	template<size_t l> using remove_batch_out_derivs = modify_batch_out_derivs_vector_impl<l, false>;

public:

	//Hyperparameters

	static constexpr size_t num_layers = sizeof...(layers);
	static constexpr size_t last_layer_index = num_layers - 1;

	//incremental loop
	template<template<size_t> class loop_body> using loop_up_layers = for_loop<0, last_layer_index - 1, 1, loop_body>;
	//decremental loop
	template<template<size_t> class loop_body> using loop_down_layers = for_loop<last_layer_index, 1, 1, loop_body>;
	template<template<size_t> class loop_body> using loop_all_layers = for_loop<0, last_layer_index, 1, loop_body>;
	//fetch a layer with a constexpr
	template<size_t l> using get_layer = get_type<l, layers...>;
	//fetch a layer activation vector with a constexpr
	template<size_t l> static typename get_layer<l>::feature_maps_vector_type& get_batch_activations()
	{
		return std::get<l, typename layers::feature_maps_vector_type...>(batch_activations);
	}
	//fetch out derivs vector
	template<size_t l> static typename get_layer<l>::feature_maps_vector_type& get_batch_out_derivs()
	{
		return std::get<l, typename layers::feature_maps_vector_type...>(batch_out_derivs);
	}


	static size_t loss_function;
	static size_t optimization_method;
	static bool use_dropout;
	static bool use_batch_learning;
	//Cannot be used with Adam
	static bool use_momentum;
	static bool use_l2_weight_decay;
	static bool include_bias_decay;

	//learning rate (should be positive)
	static float learning_rate;
	//only set if using adagrad
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

	static FeatureMap<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> input;
	static FeatureMap<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()> labels;

private:

	//used for adam
	static size_t t_adam;

	static constexpr size_t last_rbm_index = get_rbm_idx<layers...>::idx;

	//need
	static std::tuple<typename layers::feature_maps_vector_type...> batch_activations;

	//only for batches and batch norm
	static std::tuple<typename layers::feature_maps_vector_type...> batch_out_derivs;

	template<size_t l> static void dropout();

	static FeatureMap<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> error_signals();

	static typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type error_signals(typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type& batch_outputs, typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type& batch_labels);

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
	static void set_input(FeatureMap<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& new_input);

	//set labels for batch
	static void set_labels(FeatureMap<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& new_labels);

	//feed forwards
	static void discriminate();

	static void discriminate(FeatureMapVector<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& batch_input);

	//feed backwards, returns a copy of the first layer (must be deallocated)
	static FeatureMap<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> generate(FeatureMap<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& input, size_t iterations, bool use_sampling);

	//wake-sleep algorithm, only trains target layer with assumption that layers up to it have been trained
	static void pretrain(size_t markov_iterations);

	//backpropogate with selected method, returns error by loss function
	static float train();

	//backprop for a batch with selected method, returns mean error by loss function
	static float train_batch(FeatureMapVector<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& batch_input, FeatureMapVector<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& batch_labels);

	//compute the population statistics for BN networks
	static void calculate_population_statistics(FeatureMapVector<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& batch_input);

	//reset and apply gradient																  
	static void apply_gradient();

	//get current error according to loss function
	static float global_error();

	//get error for an entire batch according to loss function
	static float global_error(FeatureMapVector<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& batch_outputs, FeatureMapVector<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& batch_labels);
};

//Hyperparameter declarations

template<typename... layers> size_t NeuralNet<layers...>::loss_function = MTNN_LOSS_SQUAREERROR;
template<typename... layers> size_t NeuralNet<layers...>::optimization_method = MTNN_OPT_BACKPROP;
template<typename... layers> bool NeuralNet<layers...>::use_dropout = false;
template<typename... layers> bool NeuralNet<layers...>::use_batch_learning = false;
template<typename... layers> bool NeuralNet<layers...>::use_momentum = false;
template<typename... layers> bool NeuralNet<layers...>::use_l2_weight_decay = false;
template<typename... layers> bool NeuralNet<layers...>::include_bias_decay = false;
template<typename... layers> float NeuralNet<layers...>::learning_rate = .001f;
template<typename... layers> float NeuralNet<layers...>::minimum_divisor = .1f;
template<typename... layers> float NeuralNet<layers...>::momentum_term = .8f;
template<typename... layers> float NeuralNet<layers...>::dropout_probability = .5f;
template<typename... layers> float NeuralNet<layers...>::beta1 = .9f;
template<typename... layers> float NeuralNet<layers...>::beta2 = .99f;
template<typename... layers> float NeuralNet<layers...>::weight_decay_factor = .001f;
template<typename... layers> size_t NeuralNet<layers...>::t_adam = 0;
template<typename... layers> template<typename file_name_type> FILE* NeuralNet<layers...>::save_data_t<file_name_type>::fp = {};
template<typename... layers> template<typename file_name_type> FILE* NeuralNet<layers...>::load_data_t<file_name_type>::fp = {};
template<typename... layers> FeatureMap<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()> NeuralNet<layers...>::input = {};
template<typename... layers> FeatureMap<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> NeuralNet<layers...>::labels = {};
template<typename... layers> std::tuple<typename layers::feature_maps_vector_type...> NeuralNet<layers...>::batch_activations = std::make_tuple<typename layers::feature_maps_vector_type...>(layers::feature_maps_vector_type{ 1 }...); //init with one, will add more if necessary for batch
template<typename... layers> std::tuple<typename layers::feature_maps_vector_type...> NeuralNet<layers...>::batch_out_derivs = std::make_tuple<typename layers::feature_maps_vector_type...>(layers::feature_maps_vector_type{ 0 }...); //init with zero, will add more if necessary for batch

template<typename... layers>
inline void NeuralNet<layers...>::
setup()
{
	//Don't need this since there is implicit initialization
	/*if (!use_momentum && optimization_method != MTNN_OPT_ADAM)
	{
	loop_up_layers<delete_layer_weights_momentum>();
	loop_up_layers<delete_layer_biases_momentum>();
	}

	if (!use_hessian && optimization_method == MTNN_OPT_BACKPROP)
	{
	loop_up_layers<delete_layer_weights_aux_data>();
	loop_up_layers<delete_layer_biases_aux_data>();
	}

	if (!use_batch_normalization)
	{
	loop_up_layers<delete_layer_activations>();
	}*/
}

template<typename... layers>
template<typename file_name_type>
inline void NeuralNet<layers...>::
save_data()
{
	save_net_data<file_name_type>();
}

template<typename... layers>
template<typename file_name_type>
inline void NeuralNet<layers...>::
load_data()
{
	load_net_data<file_name_type>();
}

template<typename... layers>
inline void NeuralNet<layers...>::
set_input(FeatureMap<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()>& new_input)
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

template<typename... layers>
inline void NeuralNet<layers...>::
set_labels(FeatureMap<get_layer<sizeof...(layers)-1>::feature_maps.size(), get_layer<sizeof...(layers)-1>::feature_maps.rows(), get_layer<sizeof...(layers)-1>::feature_maps.cols()>& new_labels)
{
	for (size_t f = 0; f < labels.size(); ++f)
		for (size_t i = 0; i < labels.rows(); ++i)
			for (size_t j = 0; j < labels.cols(); ++j)
				labels[f].at(i, j) = new_labels[f].at(i, j);
}

template<typename... layers>
inline void NeuralNet<layers...>:: //todo: change output type
discriminate()
{
	loop_all_layers<reset_layer_feature_maps>();
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	loop_up_layers<feed_forwards_layer>();
}

template<typename... layers>
inline void NeuralNet<layers...>::discriminate(FeatureMapVector<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()>& batch_inputs)
{
	//adjust batch data sizes
	while (get_batch_activations<0>().size() != batch_inputs.size()) //fix sizes
	{
		if (get_batch_activations<0>().size() > batch_inputs.size())
			loop_all_layers<remove_batch_activations>();
		else
			loop_all_layers<add_batch_activations>();
	}

	//reset batch activations
	loop_all_layers<reset_layer_feature_maps>();

	get_layer<0>::feed_forwards(batch_inputs, get_batch_activations<1>());
	for_loop<1, last_layer_index - 1, 1, feed_forwards_batch_training_layer>();

}

template<typename... layers>
inline FeatureMap<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()> NeuralNet<layers...>::
generate(FeatureMap<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& input, size_t iterations, bool use_sampling)
{
	//reset all but output (or inputs?)
	loop_all_layers<reset_layer_feature_maps>();
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
	FeatureMap<get_layer<0>::feature_maps.size(), get_layer<0>::feature_maps.rows(), get_layer<0>::feature_maps.cols()> output = {};
	for (size_t f = 0; f < output.size(); ++f)
		output[f] = get_layer<0>::feature_maps[f].clone();
	return output;
}

template<typename... layers>
inline void NeuralNet<layers...>::
pretrain(size_t markov_iterations)
{
	//reset input
	loop_all_layers<reset_layer_feature_maps>();
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	loop_up_layers<feed_forwards_training_layer>();

	using target_layer = get_layer<last_layer_index>; //todo add in target layer
	if (target_layer::type == MTNN_LAYER_CONVOLUTION || target_layer::type == MTNN_LAYER_PERCEPTRONFULLCONNECTIVITY)
		target_layer::wake_sleep(learning_rate, use_dropout, markov_iterations);
}

template<typename... layers>
inline float NeuralNet<layers...>::
train()
{
	float error;
	loop_up_layers<reset_layer_feature_maps>(); //resets batch too
	for (size_t f = 0; f < get_layer<0>::feature_maps.size(); ++f)
		for (size_t i = 0; i < get_layer<0>::feature_maps.rows(); ++i)
			for (size_t j = 0; j < get_layer<0>::feature_maps.cols(); ++j)
				get_layer<0>::feature_maps[f].at(i, j) = input[f].at(i, j);
	loop_up_layers<feed_forwards_training_layer>(); //resets fm in process

	error = global_error();

	//get error signals for output and returns any layers to be skipped
	//constexpr size_t off = (loss_function == MTNN_LOSS_LOGLIKELIHOOD ? 1 : 0); todo
	auto errors = error_signals();

	//back_prop for each layer (need to get activation derivatives for output first
	get_layer<last_layer_index>::back_prop(get_layer<last_layer_index - 1>::activation, errors,
		get_batch_activations<last_layer_index>()[0], get_layer<last_layer_index>::feature_maps,
		!use_batch_learning && optimization_method == MTNN_OPT_BACKPROP, learning_rate,
		use_momentum && !use_batch_learning, momentum_term,
		use_l2_weight_decay, include_bias_decay, weight_decay_factor);
	for_loop<last_layer_index - 1, 1, 1, back_prop_layer>();

	if (!use_batch_learning && optimization_method != MTNN_OPT_BACKPROP)
		apply_gradient();

	return error;
}

template<typename... layers>
inline float NeuralNet<layers...>::
train_batch(FeatureMapVector<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()>& batch_inputs, FeatureMapVector<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& batch_labels)
{
	bool temp_batch = use_batch_learning;
	use_batch_learning = true;

	//adjust batch data sizes
	while (get_batch_activations<0>().size() != batch_labels.size()) //fix sizes
	{
		if (get_batch_activations<0>().size() > batch_labels.size())
			loop_all_layers<remove_batch_activations>();
		else
			loop_all_layers<add_batch_activations>();
	}
	while (get_batch_out_derivs<0>().size() != batch_labels.size()) //fix sizes
	{
		if (get_batch_out_derivs<0>().size() > batch_labels.size())
			loop_all_layers<remove_batch_out_derivs>();
		else
			loop_all_layers<add_batch_out_derivs>();
	}

	//reset batch activations
	loop_all_layers<reset_layer_feature_maps>();

	get_layer<0>::feed_forwards(batch_inputs, get_batch_activations<1>());
	for_loop<1, last_layer_index - 1, 1, feed_forwards_batch_training_layer>();

	float total_error = global_error(get_batch_activations<last_layer_index>(), batch_labels);

	//get error signals for output and returns any layers to be skipped
	//constexpr size_t off = (loss_function == MTNN_LOSS_LOGLIKELIHOOD ? 1 : 0); todo
	auto errors = error_signals(get_batch_activations<last_layer_index>(), batch_labels);

	//back_prop for each layer (need to get activation derivatives for output first
	get_layer<last_layer_index>::back_prop(get_layer<last_layer_index - 1>::activation, errors,
		get_batch_activations<last_layer_index>(), get_batch_out_derivs<last_layer_index>(),
		true, learning_rate, false, momentum_term,
		use_l2_weight_decay, include_bias_decay, weight_decay_factor);
	for_loop<last_layer_index - 1, 1, 1, back_prop_batch_layer>();

	apply_gradient();
	use_batch_learning = temp_batch;
	return total_error / batch_inputs.size();
}

template<typename... layers>
inline void NeuralNet<layers...>::
calculate_population_statistics(FeatureMapVector<get_type<0, layers...>::feature_maps.size(), get_type<0, layers...>::feature_maps.rows(), get_type<0, layers...>::feature_maps.cols()>& batch_inputs)
{
	//put in inputs
	get_layer<0>::feed_forwards(batch_inputs, get_batch_activations<1>());
	for_loop<1, last_layer_index - 1, 1, feed_forwards_population_statistics_layer>();
}

template<typename... layers>
inline void NeuralNet<layers...>:: //todo: maybe move apply gradient into class?
apply_gradient()
{
	if (use_l2_weight_decay && use_batch_learning)
		loop_up_layers<add_weight_decay_layer>();

	if (optimization_method == MTNN_OPT_ADAM)
		++t_adam;

	loop_up_layers<apply_gradient_layer>();
}

template<typename... layers>
inline float NeuralNet<layers...>::
global_error()
{
	float sum = 0.0f;

	if (loss_function == MTNN_LOSS_SQUAREERROR)
	{
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels[f].rows(); ++i)
				for (size_t j = 0; j < labels[f].cols(); ++j)
					sum += pow(get_layer<last_layer_index>::feature_maps[f].at(i, j) - labels[f].at(i, j), 2);
		return sum / 2;
	}
	else if (loss_function == MTNN_LOSS_LOGLIKELIHOOD)
	{
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels[f].rows(); ++i)
				for (size_t j = 0; j < labels[f].cols(); ++j)
					if (labels[f].at(i, j) > 0)
						return -log(get_layer<last_layer_index>::feature_maps[f].at(i, j));
	}
	else if (loss_function == MTNN_LOSS_CUSTOMTARGETS)
		return 0;
}

template<typename... layers>
inline float NeuralNet<layers...>::
global_error(FeatureMapVector<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& batch_outputs, FeatureMapVector<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>& batch_labels)
{
	if (loss_function == MTNN_LOSS_CUSTOMTARGETS)
		return 0;
	float sum = 0.0f;
	for (size_t in = 0; in < batch_outputs.size(); ++in)
	{
		if (loss_function == MTNN_LOSS_SQUAREERROR)
			for (size_t f = 0; f < batch_labels[in].size(); ++f)
				for (size_t i = 0; i < batch_labels[in][f].rows(); ++i)
					for (size_t j = 0; j < batch_labels[in][f].cols(); ++j)
						sum += pow(batch_outputs[in][f].at(i, j) - batch_labels[in][f].at(i, j), 2);
		else if (loss_function == MTNN_LOSS_LOGLIKELIHOOD)
			for (size_t f = 0; f < batch_labels[in].size(); ++f)
				for (size_t i = 0; i < batch_labels[in][f].rows(); ++i)
					for (size_t j = 0; j < batch_labels[in][f].cols(); ++j)
						if (batch_labels[in][f].at(i, j) > 0)
							sum += -log(batch_outputs[in][f].at(i, j));
	}
	if (loss_function == MTNN_LOSS_SQUAREERROR)
		return sum / 2;
	else if (loss_function == MTNN_LOSS_LOGLIKELIHOOD)
		return sum;
}

template<typename... layers>
template<size_t l>
inline void NeuralNet<layers...>::
dropout()
{
	using layer = get_layer<l>;
	for (size_t f = 0; f < layer::feature_maps.size(); ++f)
		for (size_t i = 0; i < layer::feature_maps.rows(); ++i)
			for (size_t j = 0; j < layer::feature_maps.cols(); ++j)
				if ((1.0f * rand()) / RAND_MAX <= dropout_probability)
					layer::feature_maps[f].at(i, j) = 0;
}

template<typename... layers>
inline FeatureMap<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()> NeuralNet<layers...>::
error_signals()
{
	auto out = FeatureMap<get_type<sizeof...(layers)-1, layers...>::feature_maps.size(), get_type<sizeof...(layers)-1, layers...>::feature_maps.rows(), get_type<sizeof...(layers)-1, layers...>::feature_maps.cols()>{ 0 };
	if (loss_function == MTNN_LOSS_SQUAREERROR)
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[f].at(i, j) = get_layer<last_layer_index>::feature_maps[f].at(i, j) - labels[f].at(i, j);
	else if (loss_function == MTNN_LOSS_LOGLIKELIHOOD) //assumes next layer is softmax
	{
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[f].at(i, j) = get_layer<last_layer_index - 1>::feature_maps[f].at(i, j) - labels[f].at(i, j);
	}
	else if (loss_function == MTNN_LOSS_CUSTOMTARGETS)
		for (size_t f = 0; f < labels.size(); ++f)
			for (size_t i = 0; i < labels.rows(); ++i)
				for (size_t j = 0; j < labels.cols(); ++j)
					out[f].at(i, j) = labels[f].at(i, j);
	return out;
}

template<typename ...layers>
inline typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type NeuralNet<layers...>::
error_signals(typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type& batch_outputs, typename get_type<sizeof...(layers)-1, layers...>::feature_maps_vector_type& batch_labels)
{
	auto out = typename get_layer<last_layer_index>::feature_maps_vector_type(batch_outputs.size());
	for (size_t in = 0; in < batch_outputs.size(); ++in)
	{
		if (loss_function == MTNN_LOSS_SQUAREERROR)
			for (size_t f = 0; f < batch_labels[in].size(); ++f)
				for (size_t i = 0; i < batch_labels[in].rows(); ++i)
					for (size_t j = 0; j < batch_labels[in].cols(); ++j)
						out[in][f].at(i, j) = batch_outputs[in][f].at(i, j) - batch_labels[in][f].at(i, j);
		else if (loss_function == MTNN_LOSS_LOGLIKELIHOOD) //assumes next layer is softmax and batch_outputs are given so
		{
			for (size_t f = 0; f < batch_labels[in].size(); ++f)
				for (size_t i = 0; i < batch_labels[in].rows(); ++i)
					for (size_t j = 0; j < batch_labels[in].cols(); ++j)
						out[in][f].at(i, j) = batch_outputs[in][f].at(i, j) - batch_labels[in][f].at(i, j);
		}
		else if (loss_function == MTNN_LOSS_CUSTOMTARGETS)
			for (size_t f = 0; f < batch_labels[in].size(); ++f)
				for (size_t i = 0; i < batch_labels[in].rows(); ++i)
					for (size_t j = 0; j < batch_labels[in].cols(); ++j)
						out[in][f].at(i, j) = batch_labels[in][f].at(i, j);
	}
	return out;
}