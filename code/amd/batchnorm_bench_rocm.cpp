#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "tensor.h"
#include "miopen_helper.h"
#include "batchnorm_problems.h"


/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/conv_bench

To run inference mode, use the following command:

bin/conv_bench inference


To change the precision for training/inference, use:

bin/conv_bench train <precision>
bin/conv_bench inference <precision>

Supported precision types:

For Maxwell GPUS: 
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

// T is used as the data type for inputs, weights and outputs. 
template <typename T>
class miopenBN {
    TensorDescriptor4d<T> x_desc_;
    TensorDescriptor4d<T> bn_scale_bias_mean_var_desc_;

    int num_repeats_;

    int inference;
    double exponential_average_factor;
    double epsilon;

    Tensor<float> running_mean_;
    Tensor<float> running_var_;
    Tensor<float> save_mean_;
    Tensor<float> save_var_;

    float alpha_ = 1.f;
    float beta_  = 0.f;

    miopenBatchNormMode_t bnMode = miopenBNSpatial;
    MIOpenHandle miopen_handle_;

public:

    miopenBN(int w, int h, int c, int n, int num_features,
            double momentum, double eps,  int inference)
            :
        miopen_handle_(),
        x_desc_(n, c, h, w),
        bn_scale_bias_mean_var_desc_(1, num_features, 1, 1)
    {

        assert(c == num_features && "num_features doesn't match number of input channels");
        
        inference = inference;
        exponential_average_factor = momentum;
        epsilon = eps;

        std::vector<int> dims_mean_var_{num_features};
        running_mean_ = zeros<float>(dims_mean_var_);
        running_var_ = fill<float>(dims_mean_var_, 1); 

        if (!inference) {
            save_mean_ = Tensor<float>(dims_mean_var_);
            save_var_ = Tensor<float>(dims_mean_var_);
        }

    }


    void forward(Tensor<T> x, Tensor<T> scale, Tensor<T> bias, Tensor<T> y) {

        // Batchnorm2d forward.
        if (!inference) {
            
            CHECK_MIOPEN_ERROR(miopenBatchNormalizationForwardTraining(miopen_handle_.handle(),
                                                                       bnMode,
                                                                       &alpha_,
                                                                       &beta_,
                                                                       x_desc_.desc(),
                                                                       x.begin(),
                                                                       x_desc_.desc(),
                                                                       y.begin(),
                                                                       bn_scale_bias_mean_var_desc_.desc(),
                                                                       scale.begin(),
                                                                       bias.begin(),
                                                                       exponential_average_factor,
                                                                       running_mean_.begin(),
                                                                       running_var_.begin(),
                                                                       epsilon,
                                                                       save_mean_.begin(),
                                                                       save_var_.begin())); 

        }
        else {
            CHECK_MIOPEN_ERROR(miopenBatchNormalizationForwardInference(miopen_handle_.handle(),
                                                                        bnMode,
                                                                        &alpha_,
                                                                        &beta_, 
                                                                        x_desc_.desc(),
                                                                        x.begin(),
                                                                        x_desc_.desc(),
                                                                        y.begin(),
                                                                        bn_scale_bias_mean_var_desc_.desc(),
                                                                        scale.begin(),
                                                                        bias.begin(),
                                                                        running_mean_.begin(),
                                                                        running_var_.begin(),
                                                                        epsilon));

        }
    }

    void backward(Tensor<T> x, Tensor<T> scale, Tensor<T> bias, Tensor<T> dY,
                  Tensor<T> dX, Tensor<T> dScale, Tensor<T> dBias) {

        CHECK_MIOPEN_ERROR(miopenBatchNormalizationBackward(miopen_handle_.handle(),
                                                            bnMode,
                                                            &alpha_,
                                                            &beta_, 
                                                            &alpha_,
                                                            &beta_, 
                                                            x_desc_.desc(),
                                                            x.begin(),
                                                            x_desc_.desc(),
                                                            dY.begin(),
                                                            x_desc_.desc(),
                                                            dX.begin(),
                                                            bn_scale_bias_mean_var_desc_.desc(),
                                                            scale.begin(),
                                                            dScale.begin(),
                                                            dBias.begin(),
                                                            epsilon,
                                                            save_mean_.begin(),
                                                            save_var_.begin()));

    }

};

template <typename T>
std::tuple<int, int> time_batchnorm(
         int w, int h, int c, int n,
         int num_features, double momentum, double eps,
         int num_repeats,
         int inference
        ) {

    miopenBN<T> batchnorm(w, h, c, n, num_features, momentum, eps, inference);

    // Allocate memory for weight(scale) and bias
    auto weight_bias_dims = std::vector<int>{num_features};
    auto weight = rand<T>(weight_bias_dims);
    auto bias = rand<T>(weight_bias_dims);

    // Allocate memory for input
    auto input_dims = std::vector<int>{w, h, c, n};
    auto input = rand<T>(input_dims);

    // Allocate memory for output tensor
    auto output = zeros<T>(input_dims);


    //Warm up
    batchnorm.forward(input, weight, bias, output);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        batchnorm.forward(input, weight, bias, output);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    int bwd_time = 0;

    if (!inference) {
        // Allocate memory for backward pass
        auto grad_output = rand<T>(input_dims);
        auto grad_input = zeros<T>(input_dims);
        auto grad_scale = zeros<T>(weight_bias_dims);
        auto grad_bias = zeros<T>(weight_bias_dims);

        // Warm up backward
        batchnorm.backward(input, weight, bias, grad_output, grad_input, grad_scale, grad_bias);

        hipDeviceSynchronize();
        start = std::chrono::steady_clock::now();

        for (int i = 0; i < num_repeats; ++i) {
            // Backward pass
            batchnorm.backward(input, weight, bias, grad_output, grad_input, grad_scale, grad_bias);
        }

        hipDeviceSynchronize();
        end = std::chrono::steady_clock::now();

        bwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    }

    return std::tuple<int, int>(fwd_time, bwd_time);

}

int main(int argc, char **argv) {

    int num_repeats = 300;

    int inference = 0;

    if (argc > 1) {
        std::string inf = "inference";
        inference = argv[1] == inf ? 1 : 0;
    }


    std::string precision = "float";
    if (argc > 2) {
        precision = argv[2];
    }


    hipFree(0);


    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(132) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "   w      h      c      n    num_features   momentum    eps    precision  fwd_time (usec)  ";

    if (!inference) {
        std::cout << "bwd_time (usec)  ";
        std::cout << "total_time (usec)";
    }

    std::cout << std::endl;

    std::cout << std::setfill('-') << std::setw(132) << "-" << std::endl;
    std::cout << std::setfill(' ');

    for (const auto &problem : (inference ? inference_server_set : training_set)) {

        // Input parameters
        int n, c, w, h;

        // BatchNorm parameters
        int num_features;
        double momentum, eps;

        std::tie(w, h, c, n, num_features, momentum, eps) = problem;


        int fwd_time, bwd_time;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

        if (precision == "float") {
            std::tie(fwd_time, bwd_time) =
                time_batchnorm<float>(w, h, c, n, num_features, momentum, eps, num_repeats, inference);
        } else {
            throw std::runtime_error(ss.str());
        }

        std::cout << std::setw(5) << w;
        std::cout << std::setw(7) << h;
        std::cout << std::setw(7) << c;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(15) << num_features;
        std::cout << std::setw(11) << momentum;
        std::cout << std::setw(7) << eps;
        std::cout << std::setw(10) << precision;
        std::cout << std::setw(15) << std::setprecision(7) << fwd_time;



        if (!inference) {
            std::cout << std::setw(22) << std::setprecision(7) << bwd_time;
            std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_time;
        }

        std::cout << std::endl;
    }

    return 0;

}
