
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

    Point2f tablePos;
    double expectedRadiusPx = 0;
    double foregroundFraction = 0;
    double edgeSupport = 0;
    double score = 0;
    int predictedNumber = -1;
};

struct GroundTruthBall
{
    int number = 0; // 1-15 or 0 for the white ball
    Point2f tablePos;
};

struct EvaluationResult
{
    int truePositives = 0, falsePositives = 0, falseNegatives = 0;
    vector<double> errorsMm;
    vector<Point2f> errorVectors;
    vector<int> detectionToTruth;
};

struct BallFeatures
{
    double whiteFraction = 0;  // fraction of ball that is white (stripes and cue ball would have it high)
    Vec3d colour;              // median (L, a, b) of the non-white pixels
};

struct BallClassifier
{
    double whiteChromaThreshold = 0; // pixels below this chroma (and bright) count as white
    vector<int> numbers; //ball numbers of each template
    vector<BallFeatures> templates; // one template per ball, from AllBalls
};

struct RecognitionResult
{
    int correct = 0, classified = 0, truthCount = 0;
    Mat confusion = Mat::zeros(16, 16, CV_32S); // rows = true ball, cols = predicted (index 0 = white)
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
void drawCandidates(const Mat& image, const vector<BallCandidate>& candidates, const string& name, const string& stageName);
Mat makeEdgeImage(const Mat& image, const TableGeometry& g);
double foregroundFractionInCircle(const Mat& mask, const Point2f& centre, double radius);
double edgeSupportOnCircle(const Mat& edges, const Point2f& centre, double radius);
double distanceToNearestPocket(const Point2f& tablePos);
vector<BallCandidate> verifyCandidates(const vector<BallCandidate>& candidates, const Mat& image, ForegroundResult& fg, const TableGeometry& g, const string& name);
vector<GroundTruthBall> getGroundTruth(const string& name);
EvaluationResult evaluateDetections(const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const Mat& rectified, const string& name);
void printEvaluation(const string& label, const EvaluationResult& r);
double chromaOf(const Vec3b& lab);
BallFeatures extractFeatures(const Mat& lab, const Point2f& centre, double radiusPx, double whiteChromaThreshold);
BallClassifier trainClassifier(const Mat& image, const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const EvaluationResult& match);
double featureDistance(const BallFeatures& a, const BallFeatures& b);
void classifyBalls(vector<BallCandidate>& balls, const Mat& image, const BallClassifier& classifier, const string& name);
void evaluateRecognition(const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const EvaluationResult& match, RecognitionResult& result, const Mat& rectified, const string& name);
void printRecognition(const string& label, const RecognitionResult& r, bool showConfusion);

void MyApplication()
{
    utils::fs::createDirectories(OUTPUT_FOLDER);

    // We know that the camera is fixed so we only need to build geometry once
    TableGeometry geometry = makeTableGeometry();
    cout << "Expected ball radius: " << geometry.minRadiusPx << " px (far cushion) to "
    << geometry.maxRadiusPx << " px (near cushion)" << endl;

    EvaluationResult total; // for step 6. summed over images
    BallClassifier classifier;
    RecognitionResult recognitionTotal;

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

        // Step 4: verification of circles
        vector<BallCandidate> balls = verifyCandidates(candidates, image, fg, geometry, name);
        cout << name << ": " << balls.size() << " balls after verification" << endl;

        // Step 6: Evaluation against GT
        vector<GroundTruthBall> truth = getGroundTruth(name);
        EvaluationResult r = evaluateDetections(balls, truth, rectified, name);
        printEvaluation(name, r);

        if (name == "AllBalls")
            classifier = trainClassifier(image, balls, truth, r);
        classifyBalls(balls, image, classifier, name);

        RecognitionResult rec;
        evaluateRecognition(balls, truth, r, rec, rectified, name);
        printRecognition(name, rec, false);
        if (name != "AllBalls")
        {
            recognitionTotal.correct += rec.correct;
            recognitionTotal.classified += rec.classified;
            recognitionTotal.truthCount += rec.truthCount;
            recognitionTotal.confusion += rec.confusion;
        }

        // overall total excluding allballs
        if (name != "AllBalls")
        {
            total.truePositives += r.truePositives;
            total.falsePositives += r.falsePositives;
            total.falseNegatives += r.falseNegatives;
            total.errorsMm.insert(total.errorsMm.end(), r.errorsMm.begin(), r.errorsMm.end());
            total.errorVectors.insert(total.errorVectors.end(), r.errorVectors.begin(), r.errorVectors.end());
        }
    }
    cout << endl << "==================== STAGE 1 RESULTS ====================" << endl;
    printEvaluation("ALL TEST IMAGES (TABLES 1-6)", total);

    cout << endl << "==================== STAGE 2 RESULTS ====================" << endl;
    printRecognition("ALL TEST IMAGES (Table1-6)", recognitionTotal, true);
}

/* ===========================================================================================================
    1. Table Geometry and Rectification
   ===========================================================================================================*/
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

/* ===========================================================================================================
    2. Foreground Segmentation
   ===========================================================================================================*/
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
            if (mask.at<uchar>(r, c)) values.push_back(image8u.at<uchar>(r, c));

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
    GaussianBlur(lab, lab, Size(5, 5), 0);

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

/* ===========================================================================================================
    3. Circle Candidate Detection
   ===========================================================================================================*/
/*
Purpose: finds circles in the original photo using the edges. The threshold is intentionally low because we will
filter out the bad ones later.
*/
vector<BallCandidate> findCandidates(const Mat& image, const Mat& tableMask, const TableGeometry& g, const string& name)
{
    Mat grey;
    cvtColor(image, grey, COLOR_BGR2GRAY);
    medianBlur(grey, grey, 5);

    const double CANNY_HIGH = 80;
    Mat edges;
    Canny(grey, edges, CANNY_HIGH / 2, CANNY_HIGH);
    saveStage(name, "3a_edges", edges);

    int minRadius = (int)floor(0.85 * g.minRadiusPx);
    int maxRadius = (int)ceil(1.15 * g.maxRadiusPx);
    double minDist = g.minRadiusPx;

    const double ACCUMULATOR_THRESHOLD = 15;
    vector<Vec3f> circles;
    HoughCircles(grey, circles, HOUGH_GRADIENT, 1, minDist, CANNY_HIGH, ACCUMULATOR_THRESHOLD, minRadius, maxRadius);

    Mat allowed;
    dilate(tableMask, allowed, getStructuringElement(MORPH_ELLIPSE, Size(oddAtLeast(2 * g.minRadiusPx), oddAtLeast(2 * g.minRadiusPx))));

    vector<BallCandidate> candidates;
    for (const Vec3f& c : circles)
    {
        Point centre(cvRound(c[0]), cvRound(c[1]));
        if (centre.x < 0 || centre.y < 0 || centre.x >= allowed.cols || centre.y >= allowed.rows) continue;
        if (allowed.at<uchar>(centre) == 0) continue;
        BallCandidate b;
        b.imagePos = Point2f(c[0], c[1]);
        b.radiusPx = c[2];
        candidates.push_back(b);
    }

    drawCandidates(image, candidates, name, "3b_candidates");
    return candidates;
}

/*
Purpose: Draws out the candidates onto the image and saves the image
*/
void drawCandidates(const Mat& image, const vector<BallCandidate>& candidates, const string& name, const string& stageName)
{
    Mat vis = image.clone();
    for (const BallCandidate& b : candidates)
    {
        circle(vis, b.imagePos, cvRound(b.radiusPx), Scalar(0, 0, 255), 2); // red outline on circle
        circle(vis, b.imagePos, 2, Scalar(0, 255, 255), FILLED); // Yellow Centre Dot
    }
    saveStage(name, stageName, vis);
}

/* ===========================================================================================================
    4. Candidate Verification
   ===========================================================================================================*/
/*
Purpose: Recreates the Hough Edges on a candidate, thickens edge by one pixel so circle thats off
still counts
*/
Mat makeEdgeImage(const Mat& image, const TableGeometry& g)
{
    Mat grey, edges;
    cvtColor(image, grey, COLOR_BGR2GRAY);
    medianBlur(grey, grey, 5);
    Canny(grey, edges, 40, 80);
    dilate(edges, edges, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    return edges;
}

/*
Purpose: draws a filled circle as a temporary mask, then counts the overlap with &
asks what fraction of circle is marked as non cloth
*/
double foregroundFractionInCircle(const Mat& mask, const Point2f& centre, double radius)
{
    Mat circleMask = Mat::zeros(mask.size(), CV_8U);
    circle(circleMask, centre, cvRound(radius), Scalar(255), FILLED);
    int inside = countNonZero(circleMask);
    if (inside == 0) return 0;
    return (double)countNonZero(mask & circleMask) / inside;
}

/*
Purpose: walks around the outline every 5 degrees. counts points that land on edge
*/
double edgeSupportOnCircle(const Mat& edges, const Point2f& centre, double radius)
{
    const int SAMPLES = 72;
    int onEdge = 0, valid = 0;
    for (int i = 0; i < SAMPLES; i++)
    {
        double angle = 2.0 * CV_PI * i / SAMPLES;
        int x = cvRound(centre.x + radius * cos(angle));
        int y = cvRound(centre.y + radius * sin(angle));
        if (x < 0 || y < 0 || x >= edges.cols || y >= edges.rows) continue;
        valid++;
        if (edges.at<uchar>(y, x)) onEdge++;
    }
    return valid ? (double)onEdge / valid : 0;
}

/*
Purpose: finds how far a table point is from the closest of the six pockets
*/
double distanceToNearestPocket(const Point2f& tablePos)
{
    const float W = (float)TABLE_WIDTH_MM, H = (float)TABLE_HEIGHT_MM;
    const vector<Point2f> pockets = {
        Point2f(0, 0), Point2f(W / 2, 0), Point2f(W, 0),
        Point2f(0, H), Point2f(W / 2, H), Point2f(W, H)
    };
    double best = 1e9;
    for (const Point2f& p : pockets) best = min(best, (double)norm(tablePos - p));
    return best;
}

/*
Purpose: this one attempts to remove the extra circels from step 3 and removes overlapping duplicates
*/
vector<BallCandidate> verifyCandidates(const vector<BallCandidate>& candidates, const Mat& image, ForegroundResult& fg, const TableGeometry& g, const string& name)
{
    Mat edges = makeEdgeImage(image, g);

    const double RADIUS_TOLERANCE = 0.25;
    const double MIN_FOREGROUND = 0.5;
    const double MIN_EDGE_SUPPORT = 0.6;

    vector<BallCandidate> passed;
    for (BallCandidate b : candidates)
    {
        b.tablePos = mapPoint(g.imageToTable, b.imagePos);
        b.expectedRadiusPx = expectedRadiusAtImagePoint(g, b.imagePos);

        double sizeError = fabs(b.radiusPx - b.expectedRadiusPx) / b.expectedRadiusPx;
        if (sizeError > RADIUS_TOLERANCE) continue;

        if (b.tablePos.x < 0 || b.tablePos.x > TABLE_WIDTH_MM) continue;
        if (b.tablePos.y < 0 || b.tablePos.y > TABLE_HEIGHT_MM) continue;

        b.foregroundFraction = foregroundFractionInCircle(fg.mask, b.imagePos, b.expectedRadiusPx);
        b.edgeSupport = edgeSupportOnCircle(edges, b.imagePos, b.radiusPx);
        if (b.foregroundFraction < MIN_FOREGROUND && b.edgeSupport < MIN_EDGE_SUPPORT) continue;

        const double POCKET_EXCLUSION_MM = 3 * BALL_RADIUS_MM;
        if (distanceToNearestPocket(b.tablePos) < POCKET_EXCLUSION_MM) continue;

        b.score = b.foregroundFraction + b.edgeSupport;
        passed.push_back(b);
    }

    stable_sort(passed.begin(), passed.end(), [](const BallCandidate& a, const BallCandidate& b) { return a.score > b.score; });

    vector<BallCandidate> balls;
    for (const BallCandidate& b : passed)
    {
        bool duplicate = false;
        for (const BallCandidate& kept : balls)
        {
            if (norm(b.tablePos - kept.tablePos) < BALL_RADIUS_MM) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) balls.push_back(b);
    }
    drawCandidates(image, balls, name, "4_verified");
    return balls;
}
/* ===========================================================================================================
    5. Ball Recognition
   ===========================================================================================================*/
/*
Purpose: how colourful a Lab pixel is: distance from neutral gray in the (a, b) plane. 8bit lab stores a and b
with offset of 128 so 128 means neutral
*/
double chromaOf(const Vec3b& lab)
{
    double a = lab[1] - 128.0, b = lab[2] - 128.0;
    return sqrt(a * a + b * b);
}

BallFeatures extractFeatures(const Mat& lab, const Point2f& centre, double radiusPx, double whiteChromaThreshold)
{
    const double SAMPLE_FRACTION = 0.75;
    const int WHITE_MIN_LIGHTNESS = 150;

    Mat circleMask = Mat::zeros(lab.size(), CV_8U);
    circle(circleMask, centre, cvRound(SAMPLE_FRACTION * radiusPx), Scalar(255), FILLED);

    vector<uchar> L, A, B;
    int total = 0, white = 0;
    for (int r = 0; r < lab.rows; r++)
        for (int c = 0; c < lab.cols; c++)
        {
            if (!circleMask.at<uchar>(r, c)) continue;
            const Vec3b& p = lab.at<Vec3b>(r, c);
            total++;
            if (chromaOf(p) < whiteChromaThreshold && p[0] > WHITE_MIN_LIGHTNESS) white++;
            else { L.push_back(p[0]); A.push_back(p[1]); B.push_back(p[2]); }
        }

    BallFeatures f;
    f.whiteFraction = total ? (double)white / total : 0;

    // median of each channel of non-white pixels
    auto median = [](vector<uchar>& v) -> double {
        if (v.empty()) return 0;
        nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    if (L.size() < 10)
        f.colour = Vec3d(255, 128, 128); // all white; cue ball
    else
        f.colour = Vec3d(median(L), median(A), median(B));

    return f;
}

BallClassifier trainClassifier(const Mat& image, const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const EvaluationResult& match)
{
    BallClassifier classifier;
    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);

    // white threshold from cue ball's own pixels
    for (size_t d = 0; d < balls.size(); d++)
    {
        int t = match.detectionToTruth[d];
        if (t < 0 || truth[t].number != 0) continue;
        vector<double> chroma;
        Mat circleMask = Mat::zeros(lab.size(), CV_8U);
        circle(circleMask, balls[d].imagePos, cvRound(0.75 * balls[d].expectedRadiusPx), Scalar(255), FILLED);
        for (int r = 0; r < lab.rows; r++)
            for (int c = 0; c < lab.cols; c++)
                if (circleMask.at<uchar>(r, c)) chroma.push_back(chromaOf(lab.at<Vec3b>(r, c)));
        stable_sort(chroma.begin(), chroma.end());
        classifier.whiteChromaThreshold = chroma[(size_t)(0.99 * (chroma.size() - 1))];
    }
    cout << "Learned white chroma threshold = " << classifier.whiteChromaThreshold << endl;

    // one template per ball
    for (size_t d = 0; d < balls.size(); d++)
    {
        int t = match.detectionToTruth[d];
        if (t < 0) continue;
        classifier.numbers.push_back(truth[t].number);
        classifier.templates.push_back(extractFeatures(lab, balls[d].imagePos, balls[d].expectedRadiusPx, classifier.whiteChromaThreshold));
    }
    cout << " Trained " << classifier.templates.size() << " ball templates" << endl;
    return classifier;
}

double featureDistance(const BallFeatures& a, const BallFeatures& b)
{
    const double WHITE_WEIGHT = 100;
    return norm(a.colour - b.colour) + WHITE_WEIGHT * fabs(a.whiteFraction - b.whiteFraction);
}

void classifyBalls(vector<BallCandidate>& balls, const Mat& image, const BallClassifier& classifier, const string& name)
{
    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);

    // every (ball, template) pair and how different they look
    struct Option { int ball, templ; double dist; };
    vector<Option> options;
    for (int b = 0; b < (int)balls.size(); b++)
    {
        BallFeatures f = extractFeatures(lab, balls[b].imagePos, balls[b].expectedRadiusPx, classifier.whiteChromaThreshold);
        for (int t = 0; t < (int)classifier.templates.size(); t++)
            options.push_back({ b, t, featureDistance(f, classifier.templates[t]) });
    }
    stable_sort(options.begin(), options.end(), [](const Option& a, const Option& b) { return a.dist < b.dist; });

    // most confident first; each ball gets one number, each number used once
    vector<bool> templateUsed(classifier.templates.size(), false);
    for (BallCandidate& b : balls) b.predictedNumber = -1;
    for (const Option& o : options)
    {
        if (balls[o.ball].predictedNumber != -1 || templateUsed[o.templ]) continue;
        balls[o.ball].predictedNumber = classifier.numbers[o.templ];
        templateUsed[o.templ] = true;
    }
}

/* ===========================================================================================================
    6. Evaluation
   ===========================================================================================================*/
/*
Purpose: get ground truth from excel.
Corrections to the spreadsheet (documented in the report):
Table1 balls 2, 4 and 11 were entered as (column, row) - their row values (1212, 1050, 1169)
are larger than the table height (683), so they have been swapped back.
AllBalls balls 11-15: checking the labels against the numbers printed on the balls shows the
spreadsheet has them in the wrong order. Spreadsheet -> actual: (1130,334) is 13 not 11,
(1013,335) is 11 not 12, (1067,336) is 12 not 13, (1245,342) is 15 not 14, (1187,345) is 14 not 15.
This matters because AllBalls is the training image for Stage 2.
Known remaining problem: Table1 balls 3 and 7 are listed 1.4 mm apart, which is physically
impossible. The correct position of one of them is unknown, so both are kept as given and the
sanity check below reports it.
*/
vector<GroundTruthBall> getGroundTruth(const string& name)
{
    const vector<string> names = {"AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6"};
    // {x, y} for ball 1, 2, ..., 15, white
    const int data[7][16][2] = {
        // AllBalls - balls 11-15 corrected (see comment above): the spreadsheet labels for the
        // bottom row do not match the numbers printed on the balls in the image
        { {908,253},{959,250},{1013,255},{1072,256},{1132,254},{1184,262},{1244,256},{1307,253},
          {901,329},{961,330},{1013,335},{1067,336},{1130,334},{1187,345},{1245,342},{797,293} },
        // Table1
        { {1092,309},{1212,429},{1174,353},{1050,335},{1211,285},{1170,261},{1173,352},{1132,331},
          {1210,240},{1170,310},{1169,407},{1135,281},{1212,374},{1091,360},{1213,331},{29,288} },
        // Table2
        { {1103,282},{1224,620},{1121,391},{1305,310},{1046,315},{621,202},{1173,361},{1132,331},
          {725,177},{1269,185},{798,665},{438,296},{1235,53},{1090,362},{1205,399},{1169,507} },
        // Table3
        { {908,23},{1224,616},{1199,447},{29,336},{1406,305},{621,210},{1294,260},{1252,226},
          {725,177},{1270,171},{798,665},{351,249},{1233,53},{901,169},{1201,399},{29,434} },
        // Table4
        { {908,23},{1407,546},{1204,444},{-1,-1},{1406,305},{621,210},{1294,260},{1252,226},
          {724,181},{1270,171},{798,665},{351,249},{1231,58},{901,169},{1201,399},{1392,88} },
        // Table5
        { {23,535},{1407,546},{1204,444},{-1,-1},{1406,305},{25,131},{1294,260},{1252,226},
          {72,538},{1270,171},{798,665},{351,249},{-1,-1},{901,169},{1201,399},{26,608} },
        // Table6
        { {23,535},{1364,477},{1204,444},{-1,-1},{1406,305},{25,131},{1294,260},{1252,226},
          {72,538},{1270,171},{-1,-1},{351,249},{-1,-1},{901,169},{1201,399},{947,629} }
    };

    vector<GroundTruthBall> truth;
    int index = (int)(find(names.begin(), names.end(), name) - names.begin());
    if (index >= (int)names.size()) return truth;

    for (int ball = 0; ball < 16; ball++)
    {
        int x = data[index][ball][0], y = data[index][ball][1];
        if (x < 0) continue;
        GroundTruthBall t;
        t.number = (ball == 15) ? 0 : ball + 1;
        t.tablePos = Point2f((float)x, (float)y);

        // every pos should be on table
        if (x >= TABLE_WIDTH_MM || y >= TABLE_HEIGHT_MM)
            cout << "Ground Truth Warning: " << name << " ball " << t.number << " (" << x << ", " << y <<
            ") is outside the table" << endl;

        truth.push_back(t);
    }

    // check if two balls closer than one diameter
    for (size_t i = 0; i < truth.size(); i++)
        for (size_t j = i+1; j < truth.size(); j++)
            if (norm(truth[i].tablePos - truth[j].tablePos) < 2 * BALL_RADIUS_MM * 0.9)
                cout << "Ground Truth Warning: " << name << " balls " << truth[i].number << " and "
                << truth[j].number << " are only " << norm(truth[i].tablePos - truth[j].tablePos)
                << " mm apart (physically impossible)" << endl;

    return truth;
}

/*
Purpose: matches detections to Ground Truth, then counts TP / FP / FN and errors.
We also do some filtering
*/
EvaluationResult evaluateDetections(const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const Mat& rectified, const string& name)
{
    const double MATCH_TOLERANCE_MM = BALL_RADIUS_MM; //futher than one radius = not on the ball
    EvaluationResult r;

    // All possible pairs w/in tolerance, closest first
    struct Pair {int det, gt; double dist;};
    vector<Pair> pairs;
    for (int d = 0; d < (int)balls.size(); d++)
        for (int t = 0; t < (int)truth.size(); t++)
        {
            double dist = norm(balls[d].tablePos - truth[t].tablePos);
            if (dist <= MATCH_TOLERANCE_MM) pairs.push_back({d, t, dist});
        }
    stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {return a.dist < b.dist;});

    // a greedy 1-1 assignment
    vector<int> detMatch(balls.size(), -1), gtMatch(truth.size(), -1);
    for (const Pair& p : pairs)
    {
        if (detMatch[p.det] != -1 || gtMatch[p.gt] != -1) continue;
        detMatch[p.det] = p.gt;
        gtMatch[p.gt] = p.det;
        r.errorsMm.push_back(p.dist);
        r.errorVectors.push_back(balls[p.det].tablePos - truth[p.gt].tablePos);
    }

    // counts of TP, FP, FN
    r.truePositives = (int)r.errorsMm.size();
    r.falsePositives = (int)balls.size() - r.truePositives;
    r.falseNegatives = (int)truth.size() - r.truePositives;
    r.detectionToTruth = detMatch;

    // picture for my report :). green = correct (line to gt), red = fp, yellow = missed ball
    Mat vis = rectified.clone();
    int rad = (int)BALL_RADIUS_MM;
    for (int t = 0; t < (int)truth.size(); t++)
    {
        string label = truth[t].number == 0 ? "W" : to_string(truth[t].number);
        if (gtMatch[t] == -1) // missed ball
            circle(vis, truth[t].tablePos, rad, Scalar(0, 255, 255), 3);
        putText(vis, label, truth[t].tablePos + Point2f(rad, -rad), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    for (int d = 0; d < (int)balls.size(); d++)
    {
        if (detMatch[d] == -1) // FP
            circle(vis, balls[d].tablePos, rad, Scalar(0, 0, 255), 3);
        else {
            circle(vis, balls[d].tablePos, rad, Scalar(0, 255, 0), 2);
            line(vis, balls[d].tablePos, truth[detMatch[d]].tablePos, Scalar(255, 255, 255), 1);
        }
    }
    saveStage(name, "6_evaluation", vis);
    return r;
}

void evaluateRecognition(const vector<BallCandidate>& balls, const vector<GroundTruthBall>& truth, const EvaluationResult& match, RecognitionResult& result, const Mat& rectified, const string& name)
{
    Mat vis = rectified.clone();
    result.truthCount = (int)truth.size();
    for (size_t d = 0; d < balls.size(); d++)
    {
        int t = match.detectionToTruth[d];
        if (t < 0 || balls[d].predictedNumber < 0) continue;  // false positive
        int trueNumber = truth[t].number, predicted = balls[d].predictedNumber;
        result.classified++;
        result.confusion.at<int>(trueNumber, predicted)++;
        bool ok = (trueNumber == predicted);
        if (ok) result.correct++;

        // picture: green = correct, red = wrong with "predicted/true"
        auto label = [](int n) { return n == 0 ? string("W") : to_string(n); };
        string text = ok ? label(predicted) : label(predicted) + "/" + label(trueNumber);
        Scalar colour = ok ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        circle(vis, balls[d].tablePos, (int)BALL_RADIUS_MM, colour, 2);
        putText(vis, text, balls[d].tablePos + Point2f(-12, 5), FONT_HERSHEY_SIMPLEX, 0.5, colour, 2);
    }
    saveStage(name, "6_recognition", vis);
}

/*
Purpose: prints precision, recall, F1 and position errors
Precision = TP / (TP + FP) // how many were real balls
Recall = TP / (TP + FN) // how many real balls were found
*/
void printEvaluation(const string& label, const EvaluationResult& r)
{
    double precision = (r.truePositives + r.falsePositives) ? (double)r.truePositives / (r.truePositives + r.falsePositives) : 0;
    double recall = (r.truePositives + r.falseNegatives) ? (double)r.truePositives / (r.truePositives + r.falseNegatives) : 0;
    double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0;

    double meanErr = 0, rmsErr = 0, maxErr = 0;
    Point2f meanVec(0, 0);
    for (size_t i = 0; i < r.errorsMm.size(); i++)
    {
        meanErr += r.errorsMm[i];
        rmsErr += r.errorsMm[i] * r.errorsMm[i];
        maxErr = max(maxErr, r.errorsMm[i]);
        meanVec += r.errorVectors[i];
    }
    if (!r.errorsMm.empty())
    {
        meanErr /= r.errorsMm.size();
        rmsErr = sqrt(rmsErr / r.errorsMm.size());
        meanVec *= 1.0f / r.errorsMm.size();
    }

    printf("%s: TP = %d FP = %d FN = %d | Precision=%.3f Recall=%.3f F1-%.3f| Error mean = %.1fmm RMS = %.1fmm Max =%.1fmm | mean offset (x=%.1f, y=%.1f)mm\n",
            label.c_str(), r.truePositives, r.falsePositives, r.falseNegatives, precision, recall, f1, meanErr, rmsErr, maxErr, meanVec.x, meanVec.y);
}

void printRecognition(const string& label, const RecognitionResult& r, bool showConfusion)
{
    double accuracy = r.classified ? (double)r.correct / r.classified : 0;
    double endToEnd = r.truthCount ? (double)r.correct / r.truthCount : 0;
    printf("%s: recognised %d/%d correctly (accuracy=%.3f) | end-to-end %d/%d (%.3f)\n",
            label.c_str(), r.correct, r.classified, accuracy, r.correct, r.truthCount, endToEnd);
    if (!showConfusion) return;

    cout << "Confusion matrix (row = true ball, columns = predicted, W = white):" << endl << "    ";
    for (int c = 0; c < 16; c++) printf("%3s", c == 0? "W" : to_string(c).c_str());
    cout << endl;
    for (int row = 0; row < 16; row++) {
        printf("%4s ", row == 0 ? "W" : to_string(row).c_str());
        for (int c = 0; c < 16; c++)
        {
            int v = r.confusion.at<int>(row, c);
            if (v) printf("%3d", v); else printf("  .");
        }
        cout << endl;
    }

    cout << "Per ball (recall = correct / times it appeared, precision = correct / times predicted):" << endl;
    for (int n = 0; n < 16; n++)
    {
        int tp = r.confusion.at<int>(n, n);
        int rowSum = (int)sum(r.confusion.row(n))[0], colSum = (int)sum(r.confusion.col(n))[0];
        printf("  %2s: recall=%.2f (%d/%d) precision=%.2f (%d/%d)\n", n == 0 ? "W" : to_string(n).c_str(),
            rowSum ? (double)tp / rowSum : 0, tp, rowSum, colSum ? (double)tp / colSum : 0, tp, colSum);
    }
}
/* ===========================================================================================================
   7. General Utilities
   ===========================================================================================================*/
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