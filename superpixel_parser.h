//
// Copyright 2019 Graviti. All Rights Reserved.
//

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>


using AdjacentTable = std::vector<std::set<int32_t>>;
using Polygon = std::vector<cv::Point2i>;

struct Direction {
    const int32_t x;
    const int32_t y;
    Direction* next;
    Direction* last;
};

class SuperpixelParser {
public:
    SuperpixelParser();
    ~SuperpixelParser();
    std::pair<std::vector<Polygon>, AdjacentTable> generate_all_polygons(const cv::Mat& label);
    void generate_json(const cv::Mat& label,
                       const std::vector<Polygon>& polygons,
                       const AdjacentTable& table,
                       const std::string& json_filename);
    void set_total_superpixels(int32_t total_superpixels);

private:
    int32_t total_superpixels_;
    Direction* direction_0_;
    Direction* direction_1_;
    Direction* direction_2_;
    Direction* direction_3_;
    Direction* direction_4_;
    Direction* direction_5_;
    Direction* direction_6_;
    Direction* direction_7_;
    std::vector<Polygon> initiate_polygons(const cv::Mat& label);
    void generate_single_polygon(const cv::Mat& label,
                               Polygon& polygon,
                               int32_t poly_label,
                               AdjacentTable& table);
};
