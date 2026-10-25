
#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
using namespace std;
using namespace cv;

// Constant Declarations
const int TABLE_WIDTH_MM = 1434;
const int TABLE_HEIGHT_MM = 683;
const double BALL_RADIUS_MM = 21.5;
const string OUTPUT_FOLDER = "Media/Pool/Output/";
const string INPUT_FOLDER = "Media/Pool/";
const vector<string> FILE_NAMES = {
    "AllBalls.jpeg",
    "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
    "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"
};

// Struct Declarations
struct TableGeometry
{
    Mat imageToTable;
    Mat tableToImage;
    vector<Point2f> imageCorners;
    double minRadiusPx = 0;
    double maxRadiusPx = 0;
};

struct ForegroundResult
{
    Mat difference;
    Mat mask;
    double threshold = 0;
};

struct BallCandidate{
    Point2f imagePos;
    float radiusPx = 0;
};

// Function Declarations
void saveStage(const string& imageName, const string& stageName, const Mat& image);
Point2f mapPoint(const Mat& homography, const Point2f& p);
double expectedRadiusAtTablePoint(const TableGeometry& g, const Point2f& tablePoint);
double expectedRadiusAtImagePoint(const TableGeometry& g, const Point2f& imagePoint);
TableGeometry makeTableGeometry();
Mat makeTableMask(const Size& imageSize, const TableGeometry& g);
Mat rectify(const Mat& image, const TableGeometry& g);
int oddAtLeast(double value);
double otsuThresholdInsideMask(const Mat& image8u, const Mat& mask);
ForegroundResult computeForeground(const Mat& image, const Mat& tableMask, const TableGeometry& g, const string& name);
vector<BallCandidate> findCandidates(const Mat& image, const Mat& tableMask, const TableGeometry& g, const string& name);
void drawCandidates(const Mat& image, const vector<BallCandidates>& candidates, const string& name, const string& stageName);

void MyApplication()
{
    utils::fs::createDirectories(OUTPUT_FOLDER);

    // We know that the camera is fixed so we only need to build geometry once
    TableGeometry geometry = makeTableGeometry();
    cout << "Expected ball radius: " << geometry.minRadiusPx << " px (far cushion) to "
    << geometry.maxRadiusPx << " px (near cushion)" << endl;

    for (const string& filename : FILE_NAMES)
    {
        Mat image = imread(INPUT_FOLDER + filename);
        if (image.empty())
        {
            cout << "Could not open " << INPUT_FOLDER + filename << endl;
            continue;
        }
        string name = filename.substr(0, filename.find('.'));

        // Step 1: Get the table region and the rectified view
        Mat tableMask = makeTableMask(image.size(), geometry);
        Mat rectified = rectify(image, geometry);
        saveStage(name, "1_rectified", rectified);

        // Tinting everything outside the table to red to check the mask
        Mat overlay = image.clone();
        overlay.setTo(Scalar(0, 0, 255), tableMask == 0);
        addWeighted(image, 0.6, overlay, 0.4, 0, overlay);
        saveStage(name, "1_table_region", overlay);

        // Step 2: Computing the foreground
        ForegroundResult fg = computeForeground(image, tableMask, geometry, name);
        cout << name << ": Otsu Threshold = " << fg.threshold
        << ", foreground pixels = " << countNonZero(fg.mask) << endl;

        // Step 3: Hough Circle candidates
        vector<BallCandidate> candidates = findCandidates(image, tableMask, geometry, name);
        cout << name << ": " << candidates.size() << " Hough candidates" << endl;
    }
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
double expectedRadiusAtTablePoint(const TableGeometry& g, const Point2f& tablePoint)
{
    Point2f a = mapPoint(g.tableToImage, tablePoint - Point2f((float)BALL_RADIUS_MM, 0));
    Point2f b = mapPoint(g.tableToImage, tablePoint + Point2f((float)BALL_RADIUS_MM, 0));
    return 0.5 * norm(b - a);
}

/*
Purpose: for taking the hough circles in photo co-ordinates back
*/
double expectedRadiusAtImagePoint(const TableGeometry& g, const Point2f& imagePoint)
{
    return expectedRadiusAtTablePoint(g, mapPoint(g.imageToTable, imagePoint));
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

/*
Purpose: creates the table mask using convexPoly
*/
Mat makeTableMask(const Size& imageSize, const TableGeometry& g)
{
    Mat mask = Mat::zeros(imageSize, CV_8U);
    vector<Point> polygon;
    for (const Point2f& p : g.imageCorners)
        polygon.push_back(Point(cvRound(p.x), cvRound(p.y)));
    
    fillConvexPoly(mask, polygon, Scalar(255));
    return mask;
}

/*
Rectifies the image
*/
Mat rectify(const Mat& image, const TableGeometry& g)
{
    Mat rectified;
    warpPerspective(image, rectified, g.imageToTable, Size(TABLE_WIDTH_MM, TABLE_HEIGHT_MM));
    return rectified;
}

/*
Purpose: for median filter. We need odd sizes for centre pixel. If im deriving a value that isnt odd
this ensures we make it odd
*/
int oddAtLeast(double value)
{
    int k = (int)ceil(value);
    return (k % 2 == 0) ? k + 1 : k;
}

/*
Purpose: OpenCV Otsu thresholds the whole image, which would include the floor and make our
threshold wrong. I want otsu to compute over the tables pixels only.
*/
double otsuThresholdInsideMask(const Mat& image8u, const Mat& mask)
{
    vector<uchar> values;
    values.reserve(countNonZero(mask));
    for (int r = 0; r < image8u.rows; r++)
        for (int c = 0; c < image8u.cols; c++)
            if (mask.at<uchar>(r, c)) values.push_back(image8u.at<uchar>(r,c));

    Mat valuesRow(1, (int)values.size(), CV_8U, values.data());
    Mat dummy;
    return threshold(valuesRow, dummy, 0, 255, THRESH_BINARY | THRESH_OTSU);
}

/*
Purpose: finds everything on the table that is not cloth. I compare each pixel with the local
colour around it, so lighting gradient across the table wont matter. I return a difference map and 
a cleaned mask.
*/
ForegroundResult computeForeground(const Mat& image, const Mat& tableMask, const TableGeometry& g, const string& name)
{
    ForegroundResult result;

    // First i want to convert to Lab and blur to help with some noise
    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);
    GaussianBlur(lab, lab, Size(5,5), 0);

    // now we calculate the local cloth bg. a median on a window about 2 ball diameters
    int backgroundKernel = oddAtLeast(4.0 * g.maxRadiusPx);
    vector<Mat> labChannels, bgChannels(3);
    split(lab, labChannels);
    for (int c = 0; c < 3; c++)
        medianBlur(labChannels[c], bgChannels[c], backgroundKernel);

    // color difference from bg. sqrt(da^2 + db^2 + (w * dL)^2)
    const double LIGHTNESS_WEIGHT = 0.5;
    Mat dL, dA, dB;
    subtract(labChannels[0], bgChannels[0], dL, noArray(), CV_32F);
    subtract(labChannels[1], bgChannels[1], dA, noArray(), CV_32F);
    subtract(labChannels[2], bgChannels[2], dB, noArray(), CV_32F);
    Mat diff = dA.mul(dA) + dB.mul(dB) + (LIGHTNESS_WEIGHT * LIGHTNESS_WEIGHT) * dL.mul(dL);
    sqrt(diff, diff);
    diff.setTo(0, tableMask == 0);
    result.difference = diff;
    saveStage(name, "2a_difference", diff);

    // otsu threshold on the table pixels only
    Mat diff8u;
    diff.convertTo(diff8u, CV_8U);
    result.threshold = otsuThresholdInsideMask(diff8u, tableMask);
    Mat mask = (diff8u > result.threshold) & tableMask;
    saveStage(name, "2b_threshold", mask);

    // some morphology based on the smallest ball in our image
    // open to remove specks, then close to rejoin brokem balls (stripes, highlights)
    int openSize = max(3, oddAtLeast(0.2 * g.minRadiusPx));
    int closeSize = max(3, oddAtLeast(0.3 * g.minRadiusPx));
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(openSize, openSize)));
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(closeSize, closeSize)));
    saveStage(name, "2c_morphology", mask);

    // final part is to fill the holes, then to keep only the outer boundaries and draw solid
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat filled = Mat::zeros(mask.size(), CV_8U);
    drawContours(filled, contours, -1, Scalar(255), FILLED);
    result.mask = filled;
    saveStage(name, "2d_mask", filled);

    return result;
}

vector<BallCandidate> findCandidates(const Mat& image, const Mat& tableMask, const TableGeometry& g, const string& name)
{
    Mat grey;
    cvtColor(image, grey, COLOR_BGR2GRAY);
    medianBlur(grey, grey, 5);

    const double CANNY_HIGH = 80;
    Mat edges;
    Canny(grey, edges, CANNY_HIGH/2, CANNY_HIGH);
    saveStage(name, "3a_edges", edges);

}