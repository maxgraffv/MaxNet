#include "NeuralNet.h"




NeuralNet::NeuralNet() 
    : conv_weight({4,3,3,3}), conv_bias({4}), fc_weight({2, 4*62*62}), fc_bias({2})
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    for (int i = 0; i < conv_weight.size(); ++i) conv_weight[i] = distribution(generator);
    for (int i = 0; i < conv_bias.size(); ++i) conv_bias[i] = 0.0f;
    for (int i = 0; i < fc_weight.size(); ++i) fc_weight[i] = distribution(generator);
    for (int i = 0; i < fc_bias.size(); ++i) fc_bias[i] = 0.0f;
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
    x = x.reshape({x.shape()[0], -1});
    x = linear(x, fc_weight, fc_bias);
    return softmax(x);
}

float NeuralNet::crossEntropyLoss(const Tensor& predictions, const Tensor& targets) {
    float loss = 0.0f;
    for (size_t n = 0; n < predictions.shape()[0]; ++n) {
        for (size_t i = 0; i < predictions.shape()[1]; ++i) {
            float target = (i == static_cast<int>(targets.at({n}))) ? 1.0f : 0.0f;
            float pred = predictions.at({n, i});
            loss -= target * std::log(pred + 1e-9f);
        }
    }
    return loss / predictions.shape()[0];
}

void NeuralNet::backward(const Tensor& input, const Tensor& output,
                            const Tensor& target, float lr) 
{
    // Oblicz różnicę (dy) między predykcją a targetem (softmax już był w forward)
    Tensor dy = output;
    for (size_t n = 0; n < output.shape()[0]; ++n) {
        size_t t = static_cast<int>(target.at({n}));
        dy.at({n, t}) -= 1.0f;
    }

    // Gradient dla fc_weight i fc_bias
    Tensor x = conv2d(input, conv_weight, 1, 0);
    x = relu(x);
    Tensor x_flat = x.reshape({x.shape()[0], -1});

    for (size_t i = 0; i < fc_weight.shape()[0]; ++i) {
        for (size_t j = 0; j < fc_weight.shape()[1]; ++j) {
            float grad = 0.0f;
            for (size_t n = 0; n < dy.shape()[0]; ++n) {
                grad += dy.at({n, i}) * x_flat.at({n, j});
            }
            fc_weight.at({i, j}) -= lr * grad / dy.shape()[0];
        }
    }

    for (size_t i = 0; i < fc_bias.shape()[0]; ++i) {
        float grad = 0.0f;
        for (size_t n = 0; n < dy.shape()[0]; ++n) {
            grad += dy.at({n, i});
        }
        fc_bias.at({i}) -= lr * grad / dy.shape()[0];
    }
}

void NeuralNet::train(const std::vector<Tensor>& inputs,
                      const std::vector<Tensor>& targets,
                      int epochs, float lr) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor pred = forward(inputs[i]);
            float loss = crossEntropyLoss(pred, targets[i]);
            total_loss += loss;
            backward(inputs[i], pred, targets[i], lr);
        }
        std::cout << "Epoch " << epoch+1 << "/" << epochs << " - Loss: " << total_loss / inputs.size() << std::endl;
    }
}