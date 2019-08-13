#include <ctime>
#include <iostream>
#include <string>
#include <cstdlib>
#include "MBS.h"

// #define DEBUG_YL
int main(int argc, char** argv)
{
    clock_t start,end;
    std::cout << "Time count start at reading" << std::endl;
    start = clock();
    // cv::Mat img = cv::imread("/home/graviti/下载/bear.jpg");
    cv::Mat img = cv::imread(argv[1]);
    double spSize = 300; // 越小越密集
    double alpha = 0.15; // 越大越规则
    // 时间没有明显增加

    MBS mbs;
    mbs.SetSuperpixelSize(spSize);
    mbs.SetAlpha(alpha);
    
    int spCnt = mbs.SuperpixelSegmentation(img);

    cv::Mat spVisual = mbs.Visualization(img);
    cv::Mat labels(img.rows,img.cols,CV_32SC1,mbs.GetSuperpixelLabels());
    std::cout << labels << std::endl;
    std::stringstream ss;
    ss << "/home/graviti/下载/SCALP_result_" << alpha << ".png";
    std::string filename = ss.str();
    cv::imwrite(filename, spVisual);
    end = clock();
    double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout << "Total Time: " << endtime*1000.0 << "ms" << std::endl;
    std::cout << "Size of this Image..." << std::endl;
    std::cout << img.rows << " * " << img.cols << std::endl;
    return 0;
}



