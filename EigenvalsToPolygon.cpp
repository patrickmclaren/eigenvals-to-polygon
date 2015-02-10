#include <stdio.h>
#include <stdexcept>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "EigenvalsToPolygon.hpp"

using namespace cv;

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "usage: EigenvalsToPolygon <ImagePath>" << std::endl;
    return -1;
  }

  Mat src = imread(argv[1], 1);
  if (!src.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  Mat label = isolateLabel(src);

  Mat processedImage = processImage(label);

  Mat output;
  resize(processedImage, output, Size(0, 0), 0.5, 0.5);
  processedImage.release();

  // imwrite("output.jpg", output);
  displayImage(output);

  output.release();

  return 0;
}

void displayImage(Mat img)
{
  namedWindow("Image Preview", CV_WINDOW_AUTOSIZE);
  imshow("Image Preview", img);
  waitKey(0);
}

Mat isolateLabel(Mat img)
{
  // TODO(patrick):
  // 1. convert to grayscale
  // 2. resize
  // 3. detect polygon in small image
  // 4. scale polygon back to large image
  // 5. rotate
  // 6. fix perspective
  // 7. crop
  
  ////////////////////////////////////////////////////////////////////////////
  // Detect dominant polygon
  ////////////////////////////////////////////////////////////////////////////

  Mat grayImg = Mat::zeros(img.size(), CV_8UC1);
  cvtColor(img, grayImg, CV_BGR2GRAY);

  double shrinkFactor = 0.25;
  double growthFactor = 1.0 / shrinkFactor;

  Mat smallGrayImg;
  resize(grayImg, smallGrayImg, Size(0, 0), shrinkFactor, shrinkFactor);

  grayImg.release();

  vector<Point> smallPolygon = detectPolygon(smallGrayImg);

  vector<Point> polygon;
  for (int i = 0; i < smallPolygon.size(); i++) {
    Point pt = smallPolygon[i];
    polygon.push_back(Point(pt.x * growthFactor, pt.y * growthFactor));
  }

  smallGrayImg.release();

  ////////////////////////////////////////////////////////////////////////////
  // Fix perspective
  ////////////////////////////////////////////////////////////////////////////

  RotatedRect bounds = minAreaRect(polygon);

  Mat straightImg;
  if (polygon.size() == 4) {
    vector<Point2f> old_v;
    for (int i = 0; i < 4; i++) {
      old_v.push_back(Point2f(polygon[i].x, polygon[i].y));
    }

    Point2f boundsVerts[4];
    bounds.points(boundsVerts);
    
    vector<Point2f> new_v;
    for (int i = 0; i < 4; i++) {
      new_v.push_back(Point2f(boundsVerts[i].x, boundsVerts[i].y));
    }

    Mat perspective = getPerspectiveTransform(old_v, new_v);

    straightImg = Mat::zeros(img.size(), img.type());
    warpPerspective(img, straightImg, perspective, img.size());
  } else {
    std::cout << "Couldn't fix perspective - not a quadrilateral, only ";
    std::cout << polygon.size();
    std::cout << " sides" << std::endl;

    straightImg = img;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Rotate image
  ////////////////////////////////////////////////////////////////////////////

  Mat rotation = getRotationMatrix2D(bounds.center, bounds.angle, 1);
  Mat rotatedImg = Mat::zeros(straightImg.size(), straightImg.type());
  warpAffine(straightImg, rotatedImg, rotation, straightImg.size());

  straightImg.release();

  ////////////////////////////////////////////////////////////////////////////
  // Crop image
  ////////////////////////////////////////////////////////////////////////////

  Mat label = cropImage(rotatedImg, bounds.size, bounds.center);

  return label;
}

Mat cropImage(Mat img, Size size, Point center)
{
  Mat subImg = Mat::zeros(size, img.type());
  getRectSubPix(img, size, center, subImg);

  img.release();

  return subImg;
}

Mat processImage(Mat img)
{
  Mat grayImg = Mat::zeros(img.size(), CV_8UC1);
  cvtColor(img, grayImg, CV_BGR2GRAY);

  img.release();

  Mat binaryImg = Mat::zeros(grayImg.size(), CV_8UC1);
  adaptiveThreshold(grayImg, binaryImg, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 17, 5);

  grayImg.release();

  Mat blurredImg = Mat::zeros(binaryImg.size(), CV_8UC1);
  GaussianBlur(binaryImg, blurredImg, Size(0, 0), 0.5);

  binaryImg.release();

  return blurredImg;
}

// TODO(patrick): refactor
vector<Point> detectPolygon(Mat src)
{
  if (src.channels() != 1) {
    throw std::invalid_argument("Image should only contain 1 channel");
  }
  
  Mat edges = detectEdges(src);

  Mat binaryEdges = Mat::zeros(edges.size(), CV_8UC1);
  adaptiveThreshold(edges, binaryEdges, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, 0);

  edges.release();
  
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
         
  findContours(binaryEdges, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  binaryEdges.release();
  
  vector<Point> largestContour = maximalContour(contours);

  vector<Point> hull;
  convexHull(largestContour, hull);

  return reducePolygonVertices(hull, 4, 0.05);
}

// TODO(patrick): c++ize
Mat detectEdges(Mat img)
{
  if (img.channels() != 1) {
    throw std::invalid_argument("Image should only contain 1 channel");
  }
  
  int blockSize = 9;
  int ksize     = 7;

  Mat eigenpack = Mat::zeros(img.size(), CV_32FC(6));
  cornerEigenValsAndVecs(img, eigenpack, blockSize, ksize);

  Mat edges = Mat::zeros(img.size(), CV_32FC1);
  for(int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      float lambda_1 = eigenpack.at<Vec6f>(i, j)[0];
      float lambda_2 = eigenpack.at<Vec6f>(i, j)[1];

      edges.at<float>(i,j) = 1.0 + pow(lambda_1 + lambda_2, 4.0 / 3.0);
    }
  }

  eigenpack.release();

  float cMin = 0, cMax = 0, cRange = 0;
  for(int i = 0; i < edges.rows; i++) {
    for(int j = 0; j < edges.cols; j++) {
      float curr = edges.at<float>(i,j);
      
      if(curr > cMax) {
        cMax = curr;
      } else if (curr < cMin) {
        cMin = curr;
      }
    }
  }

  cRange = cMax - cMin;

  for(int i = 0; i < edges.rows; i++) {
    for(int j = 0; j < edges.cols; j++) {
      edges.at<float>(i,j) = rintf((log(edges.at<float>(i,j)) / log(cRange)) * 255);
    }
  }

  Mat greyEdges;
  edges.convertTo(greyEdges, CV_8UC1);

  return greyEdges;
}

vector<Point> maximalContour(vector<vector<Point> > contours)
{
  vector<Point> maxContour;

  double currArea, maxArea = 0;
  for(int i = 0; i < contours.size(); i++) {
    vector<Point> contour = contours.at(i);

    if((currArea = contourArea(contour)) > maxArea) {
      maxArea = currArea;
      maxContour = contour;
    }
  }

  return maxContour;
}

vector<Point> reducePolygonVertices(vector<Point> shape, int vlimit, double sigma)
{
  vector<Point> endShape = shape;
  while(endShape.size() > vlimit) {
    double epsilon = sigma * arcLength(shape, true);
    approxPolyDP(shape, endShape, epsilon, true);

    if (shape.size() == endShape.size()) {
      break;
    } else {
      shape = endShape;
    }
  }

  return endShape;
}
