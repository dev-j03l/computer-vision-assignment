
#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
using namespace std;
using namespace cv;

// Constant Declarations
const int TABLE_WIDTH_MM = 1434;
const int TABLE_HEIGHT_MM = 683;
const double BALL_RADIUS_MM = 21.5;
const string OUTPUT_FOLDER = "MEDIA/Pool/Output";

// Function Declarations
void saveStage(const string& imageName, const string& stageName, const Mat& image);

void MyApplication()
{
}

/*
Purpose: saves any intermediate image with a consistent name, like Table1_2f_mask.png
This is to help me with making the report.
*/
void saveStage(const string& imageName, const string& stageName, const Mat& image)
{
    Mat toSave;
    if (image.depth() == CV_32F || image.depth() == CV_64F)
        normalize(image, toSave, 0, 255, NORM_MINMAX, CV_8U);
    else
        toSave = image;
    imwrite(OUTPUT_FOLDER + imageName + "_" + stageName + ".png", toSave);
}