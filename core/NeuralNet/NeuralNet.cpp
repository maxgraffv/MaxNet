#include "NeuralNet.h"




NeuralNet::NeuralNet()
{

}

Tensor NeuralNet::conv2d(const Tensor& input, const Tensor& kernel, int stride, int padding) {
    // Zakładamy input w formacie (N, C, H, W) - Batch, Channels, Height, Width
    // Zakładamy kernel w formacie (out_channels, in_channels, kH, kW)

    if (input.shape().size() != 4 || kernel.shape().size() != 4) {
        throw std::invalid_argument("Input and kernel must be 4D tensors");
    }

    int batch = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_height = input.shape()[2];
    int in_width  = input.shape()[3];

    int out_channels = kernel.shape()[0];
    int kH = kernel.shape()[2];
    int kW = kernel.shape()[3];

    int out_height = (in_height - kH + 2 * padding) / stride + 1;
    int out_width  = (in_width - kW + 2 * padding) / stride + 1;

    Tensor output({batch, out_channels, out_height, out_width});

    for (size_t n = 0; n < batch; ++n) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kH; ++kh) {
                            for (size_t kw = 0; kw < kW; ++kw) {
                                size_t ih = oh * stride + kh - padding;
                                size_t iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += input.at({n, ic, ih, iw}) * kernel.at({oc, ic, kh, kw});
                                }
                            }
                        }
                    }
                    output.at({n, oc, oh, ow}) = sum;
                }
            }
        }
    }

    return output;
}

Tensor NeuralNet::relu(const Tensor& input) {
    Tensor output = input; // input copy
    for (int i = 0; i < output.size(); ++i) {
        if (output[i] < 0.0f) {
            output[i] = 0.0f;
        }
    }
    return output;
}

Tensor NeuralNet::sigmoid(const Tensor& input) {
    Tensor output = input; // input copy
    for (int i = 0; i < output.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-output[i]));
    }
    return output;
}


Tensor NeuralNet::linear(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    Tensor output({input.shape()[0], weights.shape()[0]}); // (batch, out_features)
    for (size_t n = 0; n < input.shape()[0]; ++n) {
        for (size_t o = 0; o < weights.shape()[0]; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < weights.shape()[1]; ++i) {
                sum += input.at({n, i}) * weights.at({o, i});
            }
            output.at({n, o}) = sum + bias.at({o});
        }
    }
    return output;
}

Tensor NeuralNet::softmax(const Tensor& input) {
    Tensor output = input;
    for (size_t n = 0; n < input.shape()[0]; ++n) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < input.shape()[1]; ++i)
            max_val = std::max(max_val, input.at({n, i}));

        float sum = 0.0f;
        for (size_t i = 0; i < input.shape()[1]; ++i) {
            float val = std::exp(input.at({n, i}) - max_val);
            output.at({n, i}) = val;
            sum += val;
        }
        for (size_t i = 0; i < input.shape()[1]; ++i) {
            output.at({n, i}) /= sum;
        }
    }
    return output;
}

Tensor NeuralNet::forward(const Tensor& input) {
    Tensor x = conv2d(input, conv_weight, 1, 0);
    x = relu(x);
    x = x.reshape({x.shape()[0], -1});  // flatten
    x = linear(x, fc_weight, fc_bias);
    return softmax(x);
}