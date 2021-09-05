#ifndef LIBTORCH_YOLOV5_OD_H
#define LIBTORCH_YOLOV5_OD_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <string>
#include <thread>
#include <vector>

#include "i3d.h"
#include "object.h"

namespace od {
std::vector<torch::Tensor> nonMaxSuppression(
    torch::Tensor& preds, float scoreThresh = 0.5, float iouThresh = 0.5);

std::vector<object_t> detect(const int& h, const int& w, uint8_t* bgraData,
    std::vector<std::string>& classnames, torch::jit::script::Module& module,
    std::shared_ptr<i3d>& sptr_i3d);

void setup(std::vector<std::string>& classNames, torch::jit::Module& module, const std::string& torchscript,
    const std::string& classnames);
}
#endif //LIBTORCH_YOLOV5_OD_H
