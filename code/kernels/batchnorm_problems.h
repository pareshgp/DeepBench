// Vector saves w, h, c, n, num_features, momentum, epsilon
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> training_set = {

// ResNet
std::make_tuple(112, 112, 64, 112, 64, 0.1, 0.00001)

};

std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> inference_server_set = {

};

// Vector saves w, h, c, n, num_features, momentum, epsilon
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> inference_device_set = {
    //std::make_tuple(151, 40, 1, 1, 32, 20, 5, 8, 8, 8, 2),  ARM convolution seg faults with this kernel

};

