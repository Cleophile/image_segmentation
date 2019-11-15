//
// Copyright 2019 Graviti. All Rights Reserved.
//

#include "superpixel_parser.h"

#include <fstream>
#include <iostream>

#include <glog/logging.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
// SuperpixelParser Method
// INPUT: select superpixels & cv::Mat labels (of superpixels)

// PROCESS:
// 1. Traverse through labels ONCE and do:
// 1.1 Create an adjacent table of superpixels
// 1.2 Create an array of polygons, first and only element
// in this process is the first point of a polygon

// 2. For each element in polygon-array, start at the first point,
// record the label, and tour around the polygon with the same label
// 2.1 During the touring algorithm, eliminate points on the same line

// 3. (Optional) Douglas-Peucker Algorithm to smooth the edges

// Return a JSON object: with spix mask, superpixel polygon and adjacent table.

Direction direction_0 = {-1, 0, nullptr, nullptr};
Direction direction_1 = {-1, 1, nullptr, nullptr};
Direction direction_2 = {0, 1, nullptr, nullptr};
Direction direction_3 = {1, 1, nullptr, nullptr};
Direction direction_4 = {1, 0, nullptr, nullptr};
Direction direction_5 = {1, -1, nullptr, nullptr};
Direction direction_6 = {0, -1, nullptr, nullptr};
Direction direction_7 = {-1, -1, nullptr, nullptr};
Direction direction_empty = {0, 0, nullptr, nullptr};

SuperpixelParser::SuperpixelParser()
{
    direction_0_ = &direction_0;
    direction_1_ = &direction_1;
    direction_2_ = &direction_2;
    direction_3_ = &direction_3;
    direction_4_ = &direction_4;
    direction_5_ = &direction_5;
    direction_6_ = &direction_6;
    direction_7_ = &direction_7;

    direction_0_->next = direction_1_;
    direction_1_->next = direction_2_;
    direction_2_->next = direction_3_;
    direction_3_->next = direction_4_;
    direction_4_->next = direction_5_;
    direction_5_->next = direction_6_;
    direction_6_->next = direction_7_;
    direction_7_->next = direction_0_;

    direction_0_->last = direction_7_;
    direction_1_->last = direction_7_;
    direction_2_->last = direction_1_;
    direction_3_->last = direction_1_;
    direction_4_->last = direction_3_;
    direction_5_->last = direction_3_;
    direction_6_->last = direction_5_;
    direction_7_->last = direction_5_;
}

SuperpixelParser::~SuperpixelParser() {}

void SuperpixelParser::set_total_superpixels(int32_t total_superpixels) {
    total_superpixels_ = total_superpixels;
}

std::vector<Polygon> SuperpixelParser::initiate_polygons(const cv::Mat& label)
{
    // external label
    std::vector<Polygon> polygons = std::vector<Polygon>(total_superpixels_);
    for (int32_t i = 0; i < label.rows; ++i) {
        const int32_t* row_of_image = label.ptr<int32_t>(i);
        for (int32_t j = 0; j < label.cols; ++j) {
            int32_t current_label = row_of_image[j];
            if (polygons[current_label].empty()) {
                polygons[current_label].emplace_back(i, j);
            }
        }
    }
    return polygons;
}

void SuperpixelParser::generate_single_polygon(const cv::Mat& label,
                                             Polygon& polygon,
                                             int32_t superpixel_label,
                                             AdjacentTable& adjacent_table)
{
    Direction* previous_direction = &direction_empty;
    Direction* current_direction;
    previous_direction->last = direction_7_;
    int32_t current_label;
    cv::Point2i previous_point = polygon[0];
    for (;; previous_direction = current_direction) {
        current_direction = previous_direction->last;
        for(;;current_direction = current_direction->next) {
            cv::Point2i current_point;
            current_point.x = current_direction->x + previous_point.x;
            current_point.y = current_direction->y + previous_point.y;
            if (current_point.x < 0 || current_point.y < 0 || current_point.x >= label.rows ||
                current_point.y >= label.cols) {
                continue;
            }
            current_label = label.ptr<int32_t>(current_point.x)[current_point.y];
            if (current_label == superpixel_label) {
                if (current_point == polygon[0]) {
                    return;
                }
                if (current_direction == previous_direction) {
                    polygon.back() = current_point;
                } else {
                    polygon.push_back(current_point);
                }
                previous_point = current_point;
                break;
            }
            adjacent_table[superpixel_label].insert(current_label);
            CHECK_NE(current_direction->next, previous_direction->last) << "Illegal isolated point found: (" << previous_point.x << ", "<< previous_point.y << "), with label: " << superpixel_label;
        }
    }
}

std::pair<std::vector<Polygon>, AdjacentTable> SuperpixelParser::generate_all_polygons(
        const cv::Mat& label)
{
    std::vector<Polygon> polygons = initiate_polygons(label);
    AdjacentTable adjacent_table = AdjacentTable(total_superpixels_);
    // The first point of the polygon has been generated by the find adjacent table step
    // Thus, we can just traverse through polygons and rotate from the first point.
    for (std::size_t superpixel_label = 0; superpixel_label < polygons.size(); ++superpixel_label) {
        generate_single_polygon(label, polygons[superpixel_label], superpixel_label, adjacent_table);
    }
    return std::make_pair(polygons, adjacent_table);
}

void SuperpixelParser::generate_json(const cv::Mat& label,
                                     const std::vector<Polygon>& polygons,
                                     const AdjacentTable& adjacent_table,
                                     const std::string& json_filename)
{
    rapidjson::StringBuffer string_buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(string_buffer);
    writer.StartObject();
    writer.Key("superpixel_mask_matrix");
    writer.StartArray();
    for (int32_t j = 0; j < label.cols; ++j) {
        writer.StartArray();
        for (int32_t i = 0; i < label.rows; ++i) {
            writer.Int(label.ptr<int32_t>(i)[j]);
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.Key("superpixel_adjacent_table");
    writer.StartArray();
    for (const auto& table_element : adjacent_table) {
        writer.StartArray();
        for (auto adjacent_element : table_element) {
            writer.Int(adjacent_element);
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.Key("superpixel_all_polygons");
    writer.StartArray();
    for (const auto& current_polygon : polygons) {
        writer.StartArray();
        for (const auto& polygon_point : current_polygon) {
            writer.StartArray();
            writer.Int(polygon_point.y);
            writer.Int(polygon_point.x);
            writer.EndArray();
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.EndObject();
    std::ofstream out_file(json_filename);
    CHECK(out_file) << "Fail to open the file: " << json_filename;
    out_file << string_buffer.GetString();
    out_file.close();
}
