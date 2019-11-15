#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "MBS.h"
#include "superpixel_parser.h"

int main(int argc, char** argv)
{
    clock_t start,end;
    std::cout << "Time count start at reading" << std::endl;
    start = clock();
    // cv::Mat img = cv::imread("/home/graviti/下载/bear.jpg");
    cv::Mat image = cv::imread(argv[1]);
    cv::Mat img = image;
    // cv::fastNlMeansDenoisingColored(image,img);
    // cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1 ,0,
    //                                        -1, 5, -1,
    //                                       0, -1, 0);

    // cv::filter2D(image,img,image.depth(),kernel);
    
    if(img.empty())
    {
        std::cout << "Image not found, aborting program." << std::endl;
        return -1;
    }
    double spSize = 400; // 越小越密集
    double alpha = 0.03; // 越大越规则
    // 时间没有明显增加

    MBS mbs;
    mbs.set_superpixel_size(spSize);
    mbs.set_alpha(alpha);
    
    int spCnt = mbs.superpixel_segmentation(img);
    std::cout << "Superpixel Count: " << spCnt << std::endl;
    cv::Mat label(img.rows,img.cols,CV_32SC1,mbs.get_superpixel_labels());

    SuperpixelParser parser;
    AdjacentTable table;
    std::vector<Polygon> polygons;
    auto p = parser.generate_all_polygons(label);
    polygons=p.first;
    table=p.second;
    parser.generate_json(label, polygons, table, argv[2]);
    return 0;
}



