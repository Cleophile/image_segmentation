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
    double alpha = 0.05; // 越大越规则
    // 时间没有明显增加

    MBS mbs;
    mbs.SetSuperpixelSize(spSize);
    mbs.SetAlpha(alpha);
    
    int spCnt = mbs.SuperpixelSegmentation(img);

    cv::Mat labels(img.rows,img.cols,CV_32SC1,mbs.GetSuperpixelLabels());
    cv::watershed(img,labels);

    cv::Mat aftermath = img.clone();
    for (int i=1; i<img.rows-1; i++)
    {
        for (int j=1; j<img.cols-1; j++)
        {
            if(labels.at<int>(i,j)!= labels.at<int>(i,j-1) || labels.at<int>(i,j)!= labels.at<int>(i-1,j) || labels.at<int>(i,j)!= labels.at<int>(i+1,j) || labels.at<int>(i,j)!= labels.at<int>(i,j+1))

            {
                // is boundry
                aftermath.at<cv::Vec3b>(i,j)[0] = 255;
                aftermath.at<cv::Vec3b>(i,j)[1] = 255;
                aftermath.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }
    // std::cout << labels << std::endl;
    std::stringstream ss;
    ss << "/home/graviti/下载/SCALP_result" << ".png";
    std::string filename = ss.str();
    cv::imwrite(filename, aftermath);
    end = clock();
    double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout << "Total Time: " << endtime*1000.0 << "ms" << std::endl;
    std::cout << "Size of this Image..." << std::endl;
    std::cout << img.rows << " * " << img.cols << std::endl;
    return 0;
}



