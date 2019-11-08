// tm.wang@graviti.cn
// MBS Algorithm

#pragma once

#include "opencv2/opencv.hpp"

// Minimum Barrier Superpixel Segmentation
class MBS
{
public:
	MBS();
	~MBS();

	/************************************************************************/
	/* parameter setting functions                                          */
	/************************************************************************/
	// control compactness, small alpha leads to more compact superpixels,
	// [0-1] is fine, the default is 0.1 which is suitable for most cases
	void SetAlpha(double alpha);
	
	// set the average size of superpixels
	void SetSuperpixelSize(int32_t spSize);

	/************************************************************************/
	/* do the over-segmentation                                             */
	/************************************************************************/
	int32_t SuperpixelSegmentation(cv::Mat& image);

	/************************************************************************/
	/* utility functions                                                    */
	/************************************************************************/
	int32_t* GetSuperpixelLabels();
	cv::Mat GetSeeds();
	cv::Mat GetSuperpixelElements();
	
	// cv::Mat Visualization();
	// cv::Mat Visualization(cv::Mat& image);

private:
	void DistanceTransform_MBD(cv::Mat& image, float* seedsX, float* seedsY, 
		int32_t cnt, int32_t* labels, float* dmap, float factor, int32_t iter = 4);
	int32_t FastMBD(cv::Mat& img, int32_t* labels, int32_t spSize, int32_t outIter, int32_t inIter, 
		float alpha, float* seedsX, float* seedsY, int32_t cnt);
	void MergeComponents(int32_t* ioLabels, int32_t w, int32_t h);
	double _alpha;
	int32_t _spSize;

	int32_t* _labels;

	cv::Mat _seeds;

	int32_t _imgWidth;
	int32_t _imgHeight;
	int32_t _spCnt;

};
