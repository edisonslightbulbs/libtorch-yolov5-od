#ifndef YOLO_H
#define YOLO_H

#include <torch/script.h>

namespace yolo {
std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor& preds, float score_thresh = 0.5, float iou_thresh = 0.5)
{
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i) {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4)
            * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(
            pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0)
            continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
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

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t index = 0; index < indexes.sizes()[0] - 1; ++index) {
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

            // Filter by IOUs
            torch::Tensor ious = overlaps
                / (areas.select(0, indexes[0].item().toInt())
                    + torch::index_select(
                        areas, 0, indexes.slice(0, 1, indexes.sizes()[0]))
                    - overlaps);
            indexes = torch::index_select(indexes, 0,
                torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

}

#endif /*YOLO_H*/
