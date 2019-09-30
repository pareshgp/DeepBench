#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"
#include "cudnn_helper.h"
#include "batchnorm_problems.h"

#define USE_GET 0

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

#ifndef USE_TENSOR_CORES
#if CUDNN_MAJOR >= 7
#define USE_TENSOR_CORES 1
#else
#define USE_TENSOR_CORES 0
#endif
#endif

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
class cudnnBN {
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

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    CudnnHandle cudnn_handle_;

public:

    cudnnBN(int w, int h, int c, int n, int num_features,
            double momentum, double eps,  int inference)
            :
        cudnn_handle_()
    {

        assert(c == num_features && "num_features doesn't match number of input channels");
        
        inference = inference;
        exponential_average_factor = momentum;
        epsilon = eps;

        cudnnTensorFormat_t format;
        // For int8 inference, the supported format is NHWC
        if (std::is_same<T, uint8_t>::value) {
            format = CUDNN_TENSOR_NHWC;
        } else {
            format = CUDNN_TENSOR_NCHW;
        }

        x_desc_ = TensorDescriptor4d<T>(format, n, c, h, w);
        bn_scale_bias_mean_var_desc_ = TensorDescriptor4d<T>(format, 1, num_features, 1, 1);

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
            CHECK_CUDNN_ERROR(cudnnBatchNormalizationForwardTraining(cudnn_handle_.handle(),
                                                                     CUDNN_BATCHNORM_SPATIAL,
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
            CHECK_CUDNN_ERROR(cudnnBatchNormalizationForwardInference(cudnn_handle_.handle(),
                                                                      CUDNN_BATCHNORM_SPATIAL,
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

        CHECK_CUDNN_ERROR(cudnnBatchNormalizationBackward(cudnn_handle_.handle(),
                                                          CUDNN_BATCHNORM_SPATIAL,
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
         curandGenerator_t curand_gen,
         int inference
        ) {

    cudnnBN<T> batchnorm(w, h, c, n, num_features, momentum, eps, inference);

    // Allocate memory for weight(scale) and bias
    auto weight_bias_dims = std::vector<int>{num_features};
    auto weight = rand<T>(weight_bias_dims, curand_gen);
    auto bias = rand<T>(weight_bias_dims, curand_gen);

    // Allocate memory for input
    auto input_dims = std::vector<int>{w, h, c, n};
    auto input = rand<T>(input_dims, curand_gen);

    // Allocate memory for output tensor
    auto output = zeros<T>(input_dims);


    //Warm up
    batchnorm.forward(input, weight, bias, output);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        batchnorm.forward(input, weight, bias, output);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    int bwd_time = 0;

    if (!inference) {
        // Allocate memory for backward pass
        auto grad_output = rand<T>(input_dims, curand_gen);
        auto grad_input = zeros<T>(input_dims);
        auto grad_scale = zeros<T>(weight_bias_dims);
        auto grad_bias = zeros<T>(weight_bias_dims);

        // Warm up backward
        batchnorm.backward(input, weight, bias, grad_output, grad_input, grad_scale, grad_bias);

        cudaDeviceSynchronize();
        start = std::chrono::steady_clock::now();

        for (int i = 0; i < num_repeats; ++i) {
            // Backward pass
            batchnorm.backward(input, weight, bias, grad_output, grad_input, grad_scale, grad_bias);
        }

        cudaDeviceSynchronize();
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


#if CUDNN_MAJOR >= 6
    std::string precision;
    if (inference)
        precision = "int8";
    else
        precision = "half";
#else
    std::string precision = "float";
#endif
    if (argc > 2) {
        precision = argv[2];
    }

    // Handles to various cuda libraries, structures
    curandGenerator_t curand_gen;


    cudaFree(0);

    // Initialize curand_gen and set appropriate seed.
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);


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

    if (PAD_KERNELS && ((precision == "int8" && inference) || (USE_TENSOR_CORES && !inference)))
        std::cout << " pad_kerenels  ";

    std::cout << std::endl;

    std::cout << std::setfill('-') << std::setw(132) << "-" << std::endl;
    std::cout << std::setfill(' ');

    int pad_kernels_count = 0;

    for (const auto &problem : (inference ? inference_server_set : training_set)) {

        // Input parameters
        int n, c, w, h;

        // BatchNorm parameters
        int num_features;
        double momentum, eps;

        std::tie(w, h, c, n, num_features, momentum, eps) = problem;

        bool skip_kernel = false;
        bool need_padding = false;

// Don't know if this is required. #TODO
#if CUDNN_MAJOR >= 6
        int padded_c, padded_w, padded_h;
        int pad_value;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        if (precision == "int8") {
            pad_value = 4;
            if (c % pad_value || w % pad_value || h % pad_value) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(padded_c, pad_value);
                    pad_dim(padded_h, pad_value);
                    pad_dim(padded_w, pad_value);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#if (USE_TENSOR_CORES)
        // Tensor cores need channels to be a multiple of 8. So, added padding for some kernels.
        if (!inference) {
            pad_value = 8;
            if (c % pad_value) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(padded_c, pad_value);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#endif
#endif

        int fwd_time, bwd_time;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

#if CUDNN_MAJOR >= 6
        if (precision == "float") {
            std::tie(fwd_time, bwd_time) =
                time_batchnorm<float>(w, h, c, n, num_features, momentum, eps, num_repeats, curand_gen, inference);
        } else if (precision == "half") {
            std::tie(fwd_time, bwd_time) =
                time_batchnorm<uint16_t>(w, h, c, n, num_features, momentum, eps, num_repeats, curand_gen, inference);
        } else if ((precision == "int8") && inference) {
            if (!skip_kernel) {
                std::tie(fwd_time, bwd_time) =
                    time_batchnorm<uint8_t/*, int*/>(w, h, c, n, num_features, momentum, eps, num_repeats, curand_gen, inference);
            }  // remove check TODO
        } else {
            throw std::runtime_error(ss.str());
        }
#else
        if (precision != "float")
            throw std::runtime_error(ss.str());
        std::tie(fwd_time, bwd_time) =
            time_batchnorm<float>(w, h, c, n, num_features, momentum, eps, num_repeats, curand_gen, inference);
#endif

        std::cout << std::setw(5) << w;
        std::cout << std::setw(7) << h;
        std::cout << std::setw(7) << c;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(15) << num_features;
        std::cout << std::setw(11) << momentum;
        std::cout << std::setw(7) << eps;
        std::cout << std::setw(10) << precision;
        std::cout << std::setw(15) << std::setprecision(7);

        if (skip_kernel) {
            std::cout << "Not Supported";
        } else {
            std::cout << fwd_time;
        }

        if (PAD_KERNELS && precision == "int8" && inference) {
            std::cout << std::setw(15) <<  need_padding;
        }



        if (!inference) {
            std::cout << std::setw(22) << std::setprecision(7) << bwd_time;
            std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_time;
        }

        if (USE_TENSOR_CORES && PAD_KERNELS && !inference) {
            std::cout << std::setw(15) <<  need_padding;
        }

        std::cout << std::endl;
    }

    if (precision == "int8") {
        std::cout << " Total kernels ";
        if (PAD_KERNELS)
            std::cout << "padded: " << pad_kernels_count << std::endl;
        else
            std::cout << "skipped: " << pad_kernels_count << std::endl;

        std::cout << " Total kernels: " << inference_server_set.size() << std::endl;
    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
    return 0;

}
