
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

// Struct Declarations
struct TableGeometry
{
    Mat imageToTable;
    Mat tableToImage;
    vector<Point2f> imageCorners;
    double minRadiusPx = 0;
    double maxRadiusPx = 0;
};

// Function Declarations
void saveStage(const string& imageName, const string& stageName, const Mat& image);
Point2f mapPoint(const Mat& homography, const Point2f& p);
double expectedRadiusAtTablePoint(const TableGeometry& g, const Point2f& tablePoint);
TableGeometry makeTableGeometry();


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

/*
Purpose: i use this to send one point through a homography. Its needed cause perspectiveTransform
only accepts a list of points, so I wrap a single point in a vector of one.
*/
Point2f mapPoint(const Mat& homography, const Point2f& p)
{
    vector<Point2f> in = { p }, out;
    perspectiveTransform(in, out, homography);
    return out[0];
}

/*
Purpose: estimates how many pixels a balls radius will cover in the photo when a ball is at
a given table position
*/
double expectedRadiusAtTablePoint(const TableGeometry&. g, const Point2f& tablePoint)
{
    Point2f a = mapPoint(g.tableToImage, tablePoint - Point2f((float)BALL_RADIUS_MM, 0));
    Point2f b = mapPoint(g.tableToImage, tablePoint + Point2f((float)BALL_RADIUS_MM, 0));
    return 0.5 * norm(b - a);
}

TableGeometry makeTableGeometry()
{
    TableGeometry g;

    // These are the corners in the original image. I've moved them from
    // (row, column) to match the OpenCV Point2f (x = column, y = row)
    g.imageCorners = {
        Point2f(337.f,  35.f),    // Top Left:     row 35,  col 337
        Point2f(1570.f, 21.f),    // Top Right:    row 21,  col 1570
        Point2f(1888.f, 451.f),   // Bottom Right: row 451, col 1888
        Point2f(38.f,   472.f)    // Bottom Left:  row 472, col 38
    };

    // This is the matching corner  in the table image (1px = 1mm)
    // Same order as above. TL, TR, BR, BL
    vector<Point2f> tableCorners = {
        Point2f(0.f, 0.f),
        Point2f(TABLE_WIDTH_MM - 1.f, 0.f),
        Point2f(TABLE_WIDTH_MM - 1.f, TABLE_HEIGHT_MM - 1.f),
        Point2f(0.f, TABLE_HEIGHT_MM - 1.f)
    };

    // Homography in both directions
    g.imageToTable = getPerspectiveTransform(g.imageCorners, tableCorners);
    g.tableToImage = g.imageToTable.inv();

    // also add in the range of expected ball sizes in the image
    // our far cushion (y = 0) is the furthest, so balls look smallest there
    g.minRadiusPx = expectedRadiusAtTablePoint(g, Point2f(TABLE_WIDTH_MM / 2.f, 0.f));
    g.maxRadiusPx = expectedRadiusAtTablePoint(g, Point2f(TABLE_WIDTH_MM / 2.f, (float)TABLE_HEIGHT_MM));

    return g;
}