#pragma once

#include <fstream>
#include <string>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"

template<typename net> class NeuralNetAnalyzer
{
private:
    static float total_grad_error;
    static float original_net_error;

    static int n;

    static bool proportional;

    template<size_t l, bool biases_global> struct add_grad_error_impl
    {
    public:
        add_grad_error_impl()
        {
            using layer = typename net::template get_layer<l>;

            if (!biases_global)
            {
                using t = decltype(layer::weights_gradient_global);
                for (size_t d = 0; d < t::size(); ++d)
                {
                    for (size_t i = 0; i < t::rows(); ++i)
                    {
                        for (size_t j = 0; j < t::cols(); ++j)
                        {
                            ++n;

                            //decrement, get new error
                            layer::weights_global[d].at(i, j) -= .001f;
                            net::discriminate();
                            float adj_error = net::global_error();

                            //approximate (finite differences)
                            float appr_grad = -(adj_error - original_net_error) / .001f;
                            float grad = layer::weights_gradient_global[d].at(i, j);
                            //add to total
                            if (!proportional)
                                total_grad_error += abs(layer::weights_gradient_global[d].at(i, j) - appr_grad);
                            else
                                total_grad_error += abs((layer::weights_gradient_global[d].at(i, j) - appr_grad) / layer::weights_gradient_global[d].at(i, j));

                            //reset
                            layer::weights_global[d].at(i, j) += .001f;
                        }
                    }
                }
            }

            else
            {
                using t = decltype(layer::biases_gradient_global);
                for (size_t d = 0; d < t::size(); ++d)
                {
                    for (size_t i = 0; i < t::rows(); ++i)
                    {
                        for (size_t j = 0; j < t::cols(); ++j)
                        {
                            ++n;

                            layer::biases_global[d].at(i, j) -= .001f;
                            net::discriminate();

                            float adj_error = net::global_error();
                            float appr_grad = -(adj_error - original_net_error) / .001f;

                            if (!proportional)
                                total_grad_error += abs(layer::biases_gradient_global[d].at(i, j) - appr_grad);
                            else
                                total_grad_error += abs((layer::biases_gradient_global[d].at(i, j) - appr_grad) / layer::biases_gradient_global[d].at(i, j));

                            layer::biases_global[d].at(i, j) += .001f;
                        }
                    }
                }
            }
        }
    };

    template<size_t l, bool biases_global> struct add_hess_error_impl
    {
    public:
        add_hess_error_impl()
        {
            using layer = typename net::template get_layer<l>;

            if (!biases_global)
            {
                using t = decltype(layer::weights_hessian);
                for (size_t d = 0; d < t::size(); ++d)
                {
                    for (size_t i = 0; i < t::rows(); ++i)
                    {
                        for (size_t j = 0; j < t::cols(); ++j)
                        {
                            ++n;

                            //decrement, get new error
                            layer::weights_global[d].at(i, j) -= .001f;
                            net::discriminate();
                            float h_minus = net::global_error();

                            //reincrement, get new error
                            layer::weights_global[d].at(i, j) += .002f;
                            net::discriminate();
                            float h = net::global_error();

                            //approximate (finite differences)
                            float appr_grad = (h - 2 * original_net_error + h_minus) / (.001f * .001f);

                            //add to total
                            if (!proportional)
                                total_grad_error += abs(layer::weights_hessian[d].at(i, j) - appr_grad);
                            else
                                total_grad_error += abs((layer::weights_hessian[d].at(i, j) - appr_grad) / layer::weights_hessian[d].at(i, j));

                            //reset
                            layer::weights_global[d].at(i, j) -= .001f;
                        }
                    }
                }
            }

            else
            {
                using t = decltype(layer::biases_hessian);
                for (size_t d = 0; d < t::size(); ++d)
                {
                    for (size_t i = 0; i < t::rows(); ++i)
                    {
                        for (size_t j = 0; j < t::cols(); ++j)
                        {
                            ++n;

                            //decrement, get new error
                            layer::biases_global[d].at(i, j) -= .001f;
                            net::discriminate();
                            float h_minus = net::global_error();

                            //reincrement, get new error
                            layer::biases_global[d].at(i, j) += .002f;
                            net::discriminate();
                            float h = net::global_error();

                            //approximate (finite differences)
                            float appr_grad = (h - 2 * original_net_error + h_minus) / (.001f * .001f);

                            //add to total
                            if (!proportional)
                                total_grad_error += abs(layer::biases_hessian[d].at(i, j) - appr_grad);
                            else
                                total_grad_error += abs((layer::biases_hessian[d].at(i, j) - appr_grad) / layer::biases_hessian[d].at(i, j));

                            //reset
                            layer::biases_global[d].at(i, j) -= .001f;
                        }
                    }
                }
            }
        }
    };

    template<size_t l> using add_grad_error_w = add_grad_error_impl<l, false>;
    template<size_t l> using add_grad_error_b = add_grad_error_impl<l, true>;

    template<size_t l> using add_hess_error_w = add_hess_error_impl<l, false>;
    template<size_t l> using add_hess_error_b = add_hess_error_impl<l, true>;

public:
    //find mean gradient_global error from numerical approximation MAKE SURE INPUTS ARE NOT 0
    static std::pair<float, float> mean_gradient_error()
    {
        net::discriminate();

        proportional = false;
        total_grad_error = 0.0f;
        n = 0;
        original_net_error = net::global_error();

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_grad_error_w>(0);
#else
        typename net::template loop_all_layers<add_grad_error_w>();
#endif
        std::pair<float, float> errors{};
        errors.first = total_grad_error / n;

        total_grad_error = 0.0f;
        n = 0;

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_grad_error_b>(0);
#else
        typename net::template loop_all_layers<add_grad_error_b>();
#endif

        errors.second = total_grad_error / n;

        return errors;
    }

    //find mean hessian error from numerical approximation WARNING NOT NECESSARILY ACCURATE
    static std::pair<float, float> mean_hessian_error()
    {
        net::discriminate();

        proportional = false;
        total_grad_error = 0.0f;
        n = 0;
        original_net_error = net::global_error();

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_hess_error_w>(0);
#else
        typename net::template loop_all_layers<add_hess_error_w>();
#endif

        std::pair<float, float> errors{};
        errors.first = total_grad_error / n;

        total_grad_error = 0.0f;
        n = 0;

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_hess_error_b>(0);
#else
        typename net::template loop_all_layers<add_hess_error_b>();
#endif

        errors.second = total_grad_error / n;

        return errors;
    }

    //find mean proportional gradient_global error from numerical approximation MAKE SURE INPUTS ARE NOT 0
    static std::pair<float, float> proportional_gradient_error()
    {
        net::discriminate();

        proportional = true;
        total_grad_error = 0.0f;
        n = 0;
        original_net_error = net::global_error();

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_grad_error_w>(0);
#else
        typename net::template loop_all_layers<add_grad_error_w>();
#endif

        std::pair<float, float> errors{};
        errors.first = total_grad_error / n;

        total_grad_error = 0.0f;
        n = 0;

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_grad_error_b>(0);
#else
        typename net::template loop_all_layers<add_grad_error_b>();
#endif

        errors.second = total_grad_error / n;

        return errors;
    }

    //find mean proportional hessian error from numerical approximation WARNING NOT NECESSARILY ACCURATE
    static std::pair<float, float> proportional_hessian_error()
    {
        net::discriminate();

        proportional = true;
        total_grad_error = 0.0f;
        n = 0;
        original_net_error = net::global_error();

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_hess_error_w>(0);
#else
        typename net::template loop_all_layers<add_hess_error_w>();
#endif

        std::pair<float, float> errors{};
        errors.first = total_grad_error / n;

        total_grad_error = 0.0f;
        n = 0;

#if defined(_MSC_VER) || defined(__clang__)
        net::template loop_all_layers<add_hess_error_b>(0);
#else
        typename net::template loop_all_layers<add_hess_error_b>();
#endif

        errors.second = total_grad_error / n;

        return errors;
    }

    //update sample
    static void add_point(float value)
    {
        if (sample.size() == sample_size)
            sample.erase(sample.begin());
        sample.push_back(value);
    }

    //calculate the expected error
    static float mean_error()
    {
        float sum = 0.0f;
        for (size_t i = 0; i < sample.size(); ++i)
            sum += sample[i];
        errors.push_back(sum / sample.size());
        return sum / sample.size();
    }

    //save error data
    static void save_mean_error(std::string path)
    {
        std::ofstream file{ path };
        for (size_t i = 0; i < errors.size(); ++i)
            file << errors[i] << ',';
        file.flush();
    }

    static int sample_size;

    static std::vector<float> sample;
    static std::vector<float> errors;
};
template<typename net> std::vector<float> NeuralNetAnalyzer<net>::sample = {};
template<typename net> std::vector<float> NeuralNetAnalyzer<net>::errors = {};
template<typename net> float NeuralNetAnalyzer<net>::total_grad_error = 0.0f;
template<typename net> float NeuralNetAnalyzer<net>::original_net_error = 0.0f;
template<typename net> bool NeuralNetAnalyzer<net>::proportional = false;
template<typename net> int NeuralNetAnalyzer<net>::n = 0;
template<typename net> int NeuralNetAnalyzer<net>::sample_size = 0;
