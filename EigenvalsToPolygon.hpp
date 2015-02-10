#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv);

void displayImage(Mat img);

Mat isolateLabel(Mat img);

Mat detectEdges(Mat img);

Mat cropImage(Mat img, Size size, Point center);

Mat processImage(Mat img);

vector<Point> detectPolygon(Mat src);

vector<Point> maximalContour(vector<vector<Point> > contours);

vector<Point> reducePolygonVertices(vector<Point> shape, int vlimit, double sigma);
