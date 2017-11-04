#include <iostream>
#include <iomanip>
#include <time.h>

#include "imatrix.h"
#include "ilayer.h"
#include "neuralnet.h"
#include "neuralnetanalyzer.h"

#include "ImageReader.h"
#include "LabelReader.h"

//Output functions

void normal_line(std::string s)
{
    std::cout << s << std::endl;
}

void indented_line(std::string s)
{
    std::cout << '\t' << s << std::endl;
}

//Distortion functions

template<size_t r, size_t c, size_t kernel_r, size_t kernel_c, size_t s>
Matrix2D<float, (r - kernel_r) / s + 1, (c - kernel_c) / s + 1> convolve(Matrix2D<float, r, c>& input, Matrix2D<float, kernel_r, kernel_c>& kernel)
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

template<int rows, int cols, int kernel_size>
FeatureMap<1, rows, cols> distort(Matrix2D<float, rows, cols>& input, Matrix2D<float, kernel_size, kernel_size>& kernel, float elasticity, float max_stretch, float max_rot)
{
    //elastic map distort
    const int n = (kernel_size - 1) / 2;
    auto up_i = Matrix2D<float, rows + 2 * n, cols + 2 * n>{};
    auto up_j = Matrix2D<float, rows + 2 * n, cols + 2 * n>{};
    for (int i = 0; i < up_i.rows(); ++i)
    {
        for (int j = 0; j < up_i.cols(); ++j)
        {
            if (i > 2 * n - 1 && i < rows + 2 * n - 1 && j > 2 * n - 1 && j < cols + 2 * n - 1)
            {
                up_i.at(i, j) = 2.0f * rand() / RAND_MAX - 1;
                up_j.at(i, j) = 2.0f * rand() / RAND_MAX - 1;
            }
            else
            {
                up_i.at(i, j) = 0;
                up_j.at(i, j) = 0;
            }
        }
    }
    auto map_i = convolve<rows + 2 * n, cols + 2 * n, kernel_size, kernel_size, 1>(up_i, kernel);
    auto map_j = convolve<rows + 2 * n, cols + 2 * n, kernel_size, kernel_size, 1>(up_j, kernel);
    for (int i = 0; i < map_i.rows(); ++i)
    {
        for (int j = 0; j < map_i.cols(); ++j)
        {
            map_i.at(i, j) *= elasticity;
            map_j.at(i, j) *= elasticity;
        }
    }

    //affine
    float vertical_stretch = max_stretch * (2.0f * rand() / RAND_MAX - 1);
    float horizontal_stretch = max_stretch * (2.0f * rand() / RAND_MAX - 1);

    float angle = 3.1415926f * max_rot * (2.0f * rand() / RAND_MAX - 1) / 180;
    float sina = sin(angle);
    float cosa = cos(angle);

    int rows_mid = rows / 2;
    int cols_mid = cols / 2;

    for (int i = 0; i < map_i.rows(); ++i)
    {
        for (int j = 0; j < map_i.cols(); ++j)
        {
            map_i.at(i, j) -= vertical_stretch * (rows_mid - i) + (rows_mid - i) * (cosa - 1) + (j - cols_mid) * sina;
            map_j.at(i, j) += horizontal_stretch * (j - cols_mid) + (j - cols_mid) * (cosa - 1) - (rows_mid - i) * sina;
        }
    }

    //bilinear intrepolation
    auto output = FeatureMap<1, rows, cols>();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float desired_i = i - map_i.at(i, j);
            float desired_j = j - map_j.at(i, j);

            int int_i = (int)desired_i;
            int int_j = (int)desired_j;

            float frac_i = desired_i - int_i;
            float frac_j = desired_j - int_j;

            //get rectangle weights
            float w1 = (1.0 - frac_i) * (1.0 - frac_j);
            float w2 = (1.0 - frac_i) * frac_j;
            float w3 = frac_i * (1 - frac_j);
            float w4 = frac_i * frac_j;

            //check validity
            float v1 = (int_i > 0 && int_i < rows && int_j > 0 && int_j < cols) ? input.at(int_i, int_j) : -1;
            float v2 = (int_i > 0 && int_i < rows && int_j + 1 > 0 && int_j + 1 < cols) ? input.at(int_i, int_j + 1) : -1;
            float v3 = (int_i + 1 > 0 && int_i + 1 < rows && int_j > 0 && int_j < cols) ? input.at(int_i + 1, int_j) : -1;
            float v4 = (int_i + 1 > 0 && int_i + 1 < rows && int_j + 1 > 0 && int_j + 1 < cols) ? input.at(int_i + 1, int_j + 1) : -1;

            output[0].at(i, j) = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
        }
    }

    return output;
}

template<int rows, int cols>
FeatureMap<1, rows, cols> make_fm(Matrix2D<float, rows, cols> input)
{
    FeatureMap<1, rows, cols> out{};
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[0].at(i, j) = input.at(i, j);
    return out;
}

template<int r, int c> void print_matrix(Matrix2D<float, r, c> input, int width = 3)
{
    std::cout << std::setfill(' ');
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
            std::cout << std::setprecision(width - 2) << std::setw(width) << input.at(i, j);
        std::cout << std::endl;
    }
    std::cout << std::setfill('\0');
}

#define DEFAULT -1

//setup the network architecture
//typedef NeuralNet<InputLayer<1, 1, 29, 29>,
//    BatchNormalizationLayer<1, 1, 29, 29, MTNN_FUNC_LINEAR>,
//    PerceptronFullConnectivityLayer<2, 1, 29, 29, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    BatchNormalizationLayer<2, 1, 100, 1, MTNN_FUNC_LINEAR>,
//    PerceptronFullConnectivityLayer<3, 1, 100, 1, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    BatchNormalizationLayer<3, 1, 100, 1, MTNN_FUNC_LINEAR>,
//    PerceptronFullConnectivityLayer<4, 1, 100, 1, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    BatchNormalizationLayer<4, 1, 100, 1, MTNN_FUNC_LINEAR>,
//    PerceptronFullConnectivityLayer<5, 1, 100, 1, 1, 10, 1, MTNN_FUNC_LOGISTIC, true>,
//    OutputLayer<1, 1, 10, 1>> Net;

//typedef NeuralNet<InputLayer<1, 1, 29, 29>,
//    PerceptronFullConnectivityLayer<2, 1, 29, 29, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    PerceptronFullConnectivityLayer<3, 1, 100, 1, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    PerceptronFullConnectivityLayer<4, 1, 100, 1, 1, 100, 1, MTNN_FUNC_LOGISTIC, true>,
//    PerceptronFullConnectivityLayer<5, 1, 100, 1, 1, 10, 1, MTNN_FUNC_LOGISTIC, true>,
//    OutputLayer<1, 1, 10, 1>> Net;

//standard, boring
typedef NeuralNet<InputLayer<1, 1, 29, 29>,
    ConvolutionLayer<2, 1, 29, 29, 5, 2, 6, MTNN_FUNC_TANHLECUN, true, false>,
    ConvolutionLayer<3, 6, 13, 13, 5, 2, 50, MTNN_FUNC_TANHLECUN, true, false>,
    PerceptronFullConnectivityLayer<4, 50, 5, 5, 1, 100, 1, MTNN_FUNC_TANHLECUN, true>,
    PerceptronFullConnectivityLayer<5, 1, 100, 1, 1, 10, 1, MTNN_FUNC_TANHLECUN, true>,
    OutputLayer<6, 1, 10, 1>> Net;


typedef FeatureMap<1, 29, 29> NetInput;
typedef FeatureMap<1, 10, 1> NetOutput;

bool training = false;

int main()
{
    //get string path
    auto net_file_path = CSTRING("data//net.nn");
    using net_path_type = decltype(net_file_path);

    auto pretrain_file_path = CSTRING("data//mnist.nn");
    using pretrain_path_type = decltype(pretrain_file_path);

    Net::learning_rate = .001f;
    Net::use_batch_learning = true;
    Net::optimization_method = MTNN_OPT_ADAM;
    Net::loss_function = MTNN_LOSS_L2;
    NeuralNetAnalyzer<Net>::sample_size = 100;

    //timing variables
    float t = 0.0f;
    float e_t = 0.0f;
    float p_e_t = 0.0f;

    float mse = 1.0f;

    /*auto errors = NeuralNetAnalyzer<Net>::mean_gradient_error();
    std::cout << errors.first << ',' << errors.second << std::endl;*/

    if (training)
        normal_line("Training a new network");
    else
        normal_line("Testing prior network");

    //generate gaussian kernel for distortions
    if (training)
    {
        float sigma = 8;
        const int n = 5;
        auto gaussian = Matrix2D<float, n, n>();
        for (int i = 0; i < gaussian.rows(); ++i)
            for (int j = 0; j < gaussian.cols(); ++j)
                gaussian.at(i, j) = exp(-((i - n / 2) * (i - n / 2) + (j - n / 2) * (j - n / 2)) / (2 * sigma * sigma)) / (sigma * sigma * 2 * 3.1415926f);

        normal_line("Loading MNIST Database...");

        //load in images
        std::vector<std::pair<typename NetInput::type, int>> images(60000);
        std::vector<NetOutput> labels(10);
        ImageReader trainImgs("data//train-images.idx3-ubyte");
        trainImgs.defaultval = DEFAULT;
        LabelReader trainLbls("data//train-labels.idx1-ubyte");
        trainLbls.defaultval = DEFAULT;
        for (int i = 0; i < 60000; ++i)
        {
            int label = 0;
            for (int j = 0; j < trainLbls.current.rows(); ++j)
                if (trainLbls.current.at(j, 0) > 0)
                    label = j;
            labels[label] = make_fm<10, 1>(trainLbls.current.clone());
            images[i] = std::make_pair(trainImgs.current.clone(), label);

            trainImgs.next();
            trainLbls.next();
        }

        normal_line("Starting Training");
        for (int e = 0; e < 50; ++e)
        {
            //shuffle images
            //std::random_shuffle(images.begin(), images.end());

            /*for (int it = 0; it < 60000; ++it)
            {
                auto distorted = distort<29, 29, 5>(images[it].first, gaussian, .5, .15, 15);
                auto& label = labels[images[it].second];

                Net::set_input(distorted);
                Net::set_labels(label);

                float error = 10;

                error = Net::train();
                if (it == 0)
                    indented_line("First error = " + std::to_string(error));

                NeuralNetAnalyzer<Net>::add_point(error);

                if ((it + 1) % 50 == 0)
                {
                    Net::apply_gradient();
                }

                if ((it + 1) % 5000 == 0)
                {
                    mse = NeuralNetAnalyzer<Net>::mean_error();
                    indented_line("MSE = " + std::to_string(mse));
                }
            }*/

            //discrim on first 500
            int correct = 0;
            std::vector<int> totals(10);

            for (int i = 0; i < 500; ++i)
            {
                Net::set_input(make_fm<29, 29>(images[i].first));
                auto& test = Net::discriminate()[0];
                auto& label = labels[images[i].second][0];
                int max_i = 0;
                int max_j = 0;

                float max = test.at(0, 0);
                float max2 = label.at(0, 0);
                for (int j = 1; j < test.rows(); ++j)
                {
                    //normal_line(std::to_string(test.at(j, 0)));
                    //normal_line(std::to_string(label.at(j, 0)));
                    if (test.at(j, 0) > max)
                    {
                        max = test.at(j, 0);
                        max_i = j;
                    }
                    if (label.at(j, 0) > max2)
                    {
                        max2 = label.at(j, 0);
                        max_j = j;
                    }
                }

                ++totals[max_i];
                if (max_i == max_j)
                    ++correct;
            }
            normal_line("On running random trial of 500 got " + std::to_string(correct) + " correct. ");
            std::string out = "";
            for (int j = 0; j < totals.size(); ++j)
                out += std::to_string(j) + ": " + std::to_string(totals[j] / 500.0f) + "   ";
            indented_line("Distribution: " + out);

            for (int batches = 0; batches < 60000 / 50; ++batches)
            {
                auto batch_images = FeatureMapVector<1, 29, 29>(50);
                auto batch_labels = FeatureMapVector<1, 10, 1>(50);
                for (int i = 0; i < 50; ++i)
                {
                    batch_images[i] = distort<29, 29, 5>(images[batches * 50 + i].first, gaussian, .5, .15, 15);
                    batch_labels[i] = labels[images[batches * 50 + i].second];
                }

                float error = Net::train_batch(batch_images, batch_labels);

                if (batches == 0)
                    indented_line("First error = " + std::to_string(error));

                NeuralNetAnalyzer<Net>::add_point(error);

                if ((batches + 1) * 50 % 5000 == 0)
                {
                    mse = NeuralNetAnalyzer<Net>::mean_error();
                    indented_line("MSE = " + std::to_string(mse));
                }
            }

            if (Net::learning_rate > .00005f && (e + 1) % 2 == 0)
                Net::learning_rate *= 0.794183335f;
            //if (e == 10)
            //    net.learning_rate = .001f;

            t = clock() - t;
            p_e_t += t;
            normal_line("(training) Epoch " + std::to_string(e) + " was completed in " + std::to_string(t / CLOCKS_PER_SEC) + " seconds");
            Net::save_data<net_path_type>();
            NeuralNetAnalyzer<Net>::save_mean_error("data//mse.dat");
            t = clock();
        }
        std::vector<NetInput> training_set_images(60000);
        for (int i = 0; i < training_set_images.size(); ++i)
            training_set_images[i] = make_fm<29, 29>(images[i].first);
        Net::calculate_population_statistics(training_set_images);

        normal_line("Training was completed in " + std::to_string(p_e_t / CLOCKS_PER_SEC) + " seconds.");
        t = clock();
    }

    if (!training)
        Net::load_data<pretrain_path_type>();
    else
        Net::load_data<net_path_type>();

    normal_line("Starting Testing");

    ImageReader testImgs("data//t10k-images.idx3-ubyte");
    testImgs.defaultval = DEFAULT;
    LabelReader testLbls("data//t10k-labels.idx1-ubyte");
    testLbls.defaultval = DEFAULT;
    int correct = 0;

    std::vector<int> totals(10);

    for (int i = 0; i < 9999; ++i)
    {
        testImgs.next();
        testLbls.next();

        Net::set_input(make_fm<29, 29>(testImgs.current));
        auto& test = Net::discriminate()[0];
        int max_i = 0;
        float max = test.at(0, 0);
        for (int j = 1; j < 10; ++j)
        {
            if (test.at(j, 0) > max)
            {
                max = test.at(j, 0);
                max_i = j;
            }
        }

        ++totals[max_i];
        for (int j = 0; j < testLbls.current.rows(); ++j)
            if (testLbls.current.at(j, 0) == 1 && j == max_i)
                ++correct;

        if (i % 500 == 0 && i != 0)
        {
            normal_line("After " + std::to_string(i) + " tests, " + std::to_string(100.0f * correct / i) + "% were correct");
            std::string out = "";
            for (int j = 0; j < totals.size(); ++j)
                out += std::to_string(j) + ": " + std::to_string(totals[j] / (1.0f * i)) + "   ";
            normal_line("Distribution: " + out);
        }
    }

    normal_line("Press any key to exit");
    getchar();
    return 0;
}
