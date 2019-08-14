#include <ctime>
#include <vector>
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
    if(img.empty())
    {
        std::cout << "Image not found, aborting program." << std::endl;
        return -1;
    }
    double spSize = 300; // 越小越密集
    double alpha = 0.1; // 越大越规则
    // 时间没有明显增加

    MBS mbs;
    mbs.SetSuperpixelSize(spSize);
    mbs.SetAlpha(alpha);
    
    int spCnt = mbs.SuperpixelSegmentation(img);

    cv::Mat labels(img.rows,img.cols,CV_32SC1,mbs.GetSuperpixelLabels());
    // cv::watershed(img,labels);

    // Find watershed lines on the image
    std::vector<int> r_hist(26);
    std::vector<int> g_hist(26);
    std::vector<int> b_hist(26);

    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            cv::Vec3b colors = img.at<cv::Vec3b>(i,j);
            r_hist[colors[0]/10] += 1;
            g_hist[colors[1]/10] += 1;
            b_hist[colors[2]/10] += 1;
        }
    }
    


    auto smallest = std::min_element(std::begin(r_hist), std::end(r_hist));
    long rp = std::distance(std::begin(r_hist), smallest) * 10 + 5;

    smallest = std::min_element(std::begin(g_hist), std::end(g_hist));
    long gp = std::distance(std::begin(g_hist), smallest) * 10 + 5;

    smallest = std::min_element(std::begin(b_hist), std::end(b_hist));
    long bp = std::distance(std::begin(b_hist), smallest) * 10 + 5;

    cv::Mat aftermath = img.clone();
    for (int i=1; i<img.rows-1; i++)
    {
        for (int j=1; j<img.cols-1; j++)
        {
            if(labels.at<int>(i,j)!= labels.at<int>(i,j-1) || labels.at<int>(i,j)!= labels.at<int>(i-1,j) || labels.at<int>(i,j)!= labels.at<int>(i+1,j) || labels.at<int>(i,j)!= labels.at<int>(i,j+1))

            {
                // is boundry
                aftermath.at<cv::Vec3b>(i,j)[0] = rp;
                aftermath.at<cv::Vec3b>(i,j)[1] = gp;
                aftermath.at<cv::Vec3b>(i,j)[2] = bp;
            }
        }
    }
    // std::cout << labels << std::endl;
    std::stringstream ss;
    ss << "/home/graviti/下载/scalp_results/";
    if(argc>2)
        ss << argv[2];
    else
        ss << "scalp_result";
    ss << ".png";
    std::string filename = ss.str();
    cv::imwrite(filename, aftermath);
    end = clock();
    double endtime=(double)(end-start)/CLOCKS_PER_SEC;
    std::cout << "Total Time: " << endtime*1000.0 << "ms" << std::endl;
    std::cout << "Size of this Image..." << std::endl;
    std::cout << img.rows << " * " << img.cols << std::endl;
    return 0;
}



