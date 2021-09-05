#include <utility>

#include "io.h"
#include "od.h"

std::vector<torch::Tensor> od::nonMaxSuppression(
    torch::Tensor& preds, float scoreThresh, float iouThresh)
{
    std::vector<torch::Tensor> output;
    for (int64_t i = 0; i < preds.sizes()[0]; ++i) {
        torch::Tensor pred = preds.select(0, i);

        // filter using scores
        torch::Tensor scores = pred.select(1, 4)
            * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(
            pred, 0, torch::nonzero(scores > scoreThresh).select(1, 0));
        if (pred.sizes()[0] == 0)
            continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // compute scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple
            = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({ dets.sizes()[0] });
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1))
            * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple
            = torch::sort(dets.select(1, 4), 0, true);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);

        int count = 0;
        while (indexes.sizes()[0] > 0) {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // compute overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (int64_t index = 0; index < indexes.sizes()[0] - 1; ++index) {
                lefts[index] = std::max(dets[indexes[0]][0].item().toFloat(),
                    dets[indexes[index + 1]][0].item().toFloat());
                tops[index] = std::max(dets[indexes[0]][1].item().toFloat(),
                    dets[indexes[index + 1]][1].item().toFloat());
                rights[index] = std::min(dets[indexes[0]][2].item().toFloat(),
                    dets[indexes[index + 1]][2].item().toFloat());
                bottoms[index] = std::min(dets[indexes[0]][3].item().toFloat(),
                    dets[indexes[index + 1]][3].item().toFloat());
                widths[index] = std::max(float(0),
                    rights[index].item().toFloat()
                        - lefts[index].item().toFloat());
                heights[index] = std::max(float(0),
                    bottoms[index].item().toFloat()
                        - tops[index].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // filter IOUs
            torch::Tensor iou = overlaps
                / (areas.select(0, indexes[0].item().toInt())
                    + torch::index_select(
                        areas, 0, indexes.slice(0, 1, indexes.sizes()[0]))
                    - overlaps);
            indexes = torch::index_select(indexes, 0,
                torch::nonzero(iou <= iouThresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

void od::setup(
    std::vector<std::string>& classNames, torch::jit::script::Module& module,
    const std::string& torchscript, const std::string& classnames)
{
    module = torch::jit::load(torchscript);
    std::ifstream f(classnames);
    std::string name;
    while (std::getline(f, name)) {
        classNames.push_back(name);
    }
}

std::vector<std::pair<cv::Rect, std::vector<std::string>>> od::detect(const int& h, const int& w, uint8_t* bgraData,
    std::vector<std::string>& classnames, torch::jit::script::Module& module,
    std::shared_ptr<i3d>& sptr_i3d)
{
    std::vector<std::string> labelInfo(2);
    std::pair<cv::Rect, std::vector<std::string>> detectedObject;
    std::vector<std::pair<cv::Rect, std::vector<std::string>>> results;

    clock_t start = clock();

    cv::Mat frame, frameResized;
    frame = cv::Mat(h, w, CV_8UC4, (void*)bgraData, cv::Mat::AUTO_STEP).clone();

    // format frame for tensor input
    cv::resize(frame, frameResized, cv::Size(640, 640));
    cv::cvtColor(frameResized, frameResized, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(frameResized.data,
        { frameResized.rows, frameResized.cols, 3 }, torch::kByte);
    imgTensor = imgTensor.permute({ 2, 0, 1 });
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);

    torch::Tensor preds // preds: [?, 15120, 9]
        = module.forward({ imgTensor }).toTuple()->elements()[0].toTensor();
    std::vector<torch::Tensor> dets = od::nonMaxSuppression(preds, 0.4, 0.5);

    // show detected objects
    if (!dets.empty()) {
        for (int64_t i = 0; i < dets[0].sizes()[0]; ++i) {
            auto left = (int)(dets[0][i][0].item().toFloat() * (float)frame.cols
                / 640);
            auto top = (int)(dets[0][i][1].item().toFloat() * (float)frame.rows
                / 640);
            auto right = (int)(dets[0][i][2].item().toFloat()
                * (float)frame.cols / 640);
            auto bottom = (int)(dets[0][i][3].item().toFloat()
                * (float)frame.rows / 640);
            float score = dets[0][i][4].item().toFloat();
            int classID = dets[0][i][5].item().toInt();

            // get bounding box, class name, and confidence
            cv::Rect boundingBox = cv::Rect(left, top, (right - left), (bottom - top));
            std::string classname = classnames[classID];
            std::string confidence = cv::format("%.2f", score);

            labelInfo[0] = classname;
            labelInfo[1] = confidence;
            detectedObject.first = boundingBox;
            detectedObject.second = labelInfo;
            results.emplace_back(detectedObject);

            // use class name and confidence as label
            std::string label = classname.append(" : ").append(confidence);

            // set bounding regions to show
            cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 1);

            // set detected object class name and confidence to show
            cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, (double)(right - left) / 200, cv::Scalar(0, 255, 0), 1);
        }
    }

    // set FPS to show
    cv::putText(frame,
        "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
        cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0),
        2);

    // show image frame with bounding region, class name, confidence and FPS
    cv::imshow("i3d", frame);
    if (cv::waitKey(1) == 27) {
        sptr_i3d->stop();
    }
    return results;
}
