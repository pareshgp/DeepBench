// Vector saves w, h, c, n, num_features, momentum, epsilon
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> training_set = {

// ResNet101
std::make_tuple(112,112,64,112,64,0.1,1e-05),
std::make_tuple(56,56,64,112,64,0.1,1e-05),
std::make_tuple(56,56,256,112,256,0.1,1e-05),
std::make_tuple(56,56,128,112,128,0.1,1e-05),
std::make_tuple(28,28,128,112,128,0.1,1e-05),
std::make_tuple(28,28,512,112,512,0.1,1e-05),
std::make_tuple(28,28,256,112,256,0.1,1e-05),
std::make_tuple(14,14,256,112,256,0.1,1e-05),
std::make_tuple(14,14,1024,112,1024,0.1,1e-05),
std::make_tuple(14,14,512,112,512,0.1,1e-05),
std::make_tuple(7,7,512,112,512,0.1,1e-05),
std::make_tuple(7,7,2048,112,2048,0.1,1e-05)

// Resnet152
std::make_tuple(112,112,64,80,64,0.1,1e-05),
std::make_tuple(56,56,64,80,64,0.1,1e-05),
std::make_tuple(56,56,256,80,256,0.1,1e-05),
std::make_tuple(56,56,128,80,128,0.1,1e-05),
std::make_tuple(28,28,128,80,128,0.1,1e-05),
std::make_tuple(28,28,512,80,512,0.1,1e-05),
std::make_tuple(28,28,256,80,256,0.1,1e-05),
std::make_tuple(14,14,256,80,256,0.1,1e-05),
std::make_tuple(14,14,1024,80,1024,0.1,1e-05),
std::make_tuple(14,14,512,80,512,0.1,1e-05),
std::make_tuple(7,7,512,80,512,0.1,1e-05),
std::make_tuple(7,7,2048,80,2048,0.1,1e-05)

};


std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> inference_server_set = {

// ResNet101
std::make_tuple(112,112,64,112,64,0.1,1e-05),
std::make_tuple(56,56,64,112,64,0.1,1e-05),
std::make_tuple(56,56,256,112,256,0.1,1e-05),
std::make_tuple(56,56,128,112,128,0.1,1e-05),
std::make_tuple(28,28,128,112,128,0.1,1e-05),
std::make_tuple(28,28,512,112,512,0.1,1e-05),
std::make_tuple(28,28,256,112,256,0.1,1e-05),
std::make_tuple(14,14,256,112,256,0.1,1e-05),
std::make_tuple(14,14,1024,112,1024,0.1,1e-05),
std::make_tuple(14,14,512,112,512,0.1,1e-05),
std::make_tuple(7,7,512,112,512,0.1,1e-05),

// Resnet152
std::make_tuple(112,112,64,80,64,0.1,1e-05),
std::make_tuple(56,56,64,80,64,0.1,1e-05),
std::make_tuple(56,56,256,80,256,0.1,1e-05),
std::make_tuple(56,56,128,80,128,0.1,1e-05),
std::make_tuple(28,28,128,80,128,0.1,1e-05),
std::make_tuple(28,28,512,80,512,0.1,1e-05),
std::make_tuple(28,28,256,80,256,0.1,1e-05),
std::make_tuple(14,14,256,80,256,0.1,1e-05),
std::make_tuple(14,14,1024,80,1024,0.1,1e-05),
std::make_tuple(14,14,512,80,512,0.1,1e-05),
std::make_tuple(7,7,512,80,512,0.1,1e-05),
std::make_tuple(7,7,2048,80,2048,0.1,1e-05)
std::make_tuple(7,7,2048,112,2048,0.1,1e-05)

};

// Vector saves w, h, c, n, num_features, momentum, epsilon
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, double, double>> inference_device_set = {
    //std::make_tuple(151, 40, 1, 1, 32, 20, 5, 8, 8, 8, 2),  ARM convolution seg faults with this kernel

};

