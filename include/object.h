#ifndef OBJECT_H
#define OBJECT_H

#include <opencv2/opencv.hpp>
#include <string>

struct object_t {
    cv::Mat m_image;
    int m_imageHeight;
    int m_imageWidth;
    std::string m_label;
    cv::Rect m_bbox;
    std::string m_classname;
    std::string m_confidence;
    cv::Point m_bboxOrigin; // bottom left corner
};
#endif //OBJECT_H
