#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

/// Global variables

int H_MIN = 50;
int H_MAX = 180;
int S_MIN = 90;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;
int ERODE_NUM = 1;
int DILATE_NUM = 1;
int IndexOfBiggestContour;

RNG rng(12345);
Size size(2988,5312);
Size window_size(290,531);

string window_name = "Threshold Demo";
string const trackbarWindowName = "Trackbars";

Rect bounding_rect;
Mat src, src_hsv, dst, res, src2, dst2, Dialate2, Erode2, drawing, drawing2;

int findBiggestContour( vector< vector<Point> > contours );

void on_trackbar( int, void* )
{
    //Doing nothing here
}

void createTrackbars()
{
    namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf( TrackbarName, "H_MIN", H_MIN);
    sprintf( TrackbarName, "H_MAX", H_MAX);
    sprintf( TrackbarName, "S_MIN", S_MIN);
    sprintf( TrackbarName, "S_MAX", S_MAX);
    sprintf( TrackbarName, "V_MIN", V_MIN);
    sprintf( TrackbarName, "V_MAX", V_MAX);
    sprintf( TrackbarName, "Erode", ERODE_NUM);
    sprintf( TrackbarName, "Dilate", DILATE_NUM);

    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
    createTrackbar( "Erode", trackbarWindowName, &ERODE_NUM, 30, on_trackbar );
    createTrackbar( "Dilate", trackbarWindowName, &DILATE_NUM, 30, on_trackbar );
}

/**
 * @function main
 */
int main( int argc, char** argv )
{
    /// Load an image
    src = imread( argv[1], 1 );
    createTrackbars();
    /// Convert the BGR image to HSV
    cvtColor( src, src_hsv, CV_BGR2HSV );

    while(true)
    {
        inRange(src_hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), dst);
        Mat Erode(size, CV_8UC1);
        Erode = dst;

        for(int i=0; i < ERODE_NUM; i++ )
        {
            erode(Erode, Erode, Mat(), Point(-1, -1));
        }

        Mat Dialate(size, CV_8UC1);
        Dialate= Erode;

        for( int i=0; i<DILATE_NUM; i++)
        {
            dilate(Dialate, Dialate, Mat(), Point(-1, -1), 2);
        }

        //  bitwise_and(src, src, res ,dst);
        vector<Vec4i> hierarchy;
        vector< vector<Point> > contours_hull;

        findContours(Dialate.clone(), contours_hull, hierarchy, CV_RETR_TREE, /*CV_CLOCKWISE*/CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        int s = findBiggestContour(contours_hull);

        drawing = Mat::zeros( src.size(), CV_8UC1 );
        drawContours( drawing, contours_hull, s, Scalar(150, 150, 255), 10, 8, hierarchy, 0, Point() );
        bounding_rect = boundingRect( contours_hull[s] );
        cout << "width of rectangle: " << bounding_rect.width << endl;
        cout << "height of rectangle: " << bounding_rect.height << endl;

        rectangle(src, bounding_rect, Scalar(0, 255, 0), 1, 8, 0 );

        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly( contours_hull.size() );
        vector<Rect> boundRect( contours_hull.size() );
        vector<Point2f> center( contours_hull.size() );
        vector<float> radius( contours_hull.size() );

        for( int i = 0; i < contours_hull.size(); i++ )
        {
            approxPolyDP( Mat(contours_hull[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }
          /// Draw polygonal contour + bonding rects + circles
          /*Mat drawing3 = Mat::zeros( Dialate.size(), CV_8UC3 );
          for( int i = 0; i < contours_hull.size(); i++ )
             {
               Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
               drawContours( drawing3, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
               rectangle( drawing3, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
               circle( drawing3, center[i], (int)radius[i], color, 2, 8, 0 );
             }
            */

        //Draw rotated rect and show the size of the box
        RotatedRect box = minAreaRect(contours_hull[0]);
        Point2f vertices[4];
        box.points(vertices);

        for( int i = 0; i < 4; ++i )
        {
            line(src, vertices[i],vertices[(i+1)%4], Scalar( 0, 255, 0 ), 1, CV_AA );
        }
        Size box_size = box.size;
        int box_width = box_size.width;
        int box_height = box_size.height;
        cout << "RotatedRect Width: " << box_width << endl;
        cout << "RotatedRect Height: " << box_height << endl;

        //putText(src, to_string(box_width), Point(100, 100), FONT_HERSHEY_SIMPLEX, 10, Scalar(255, 0, 0), 4, 8, true);
        //putText(src, to_string(box_height), Point(200, 100), FONT_HERSHEY_SIMPLEX, 10, Scalar(255, 0, 0), 4, 8, true);

        resize(Erode, Erode2, window_size);
        resize(Dialate, Dialate2, window_size);
        resize(dst, dst2, window_size);
        resize(src, src2, window_size);
        resize(drawing, drawing2, window_size);

        //namedWindow( "Erode", CV_WINDOW_AUTOSIZE );
        //imshow("Erode", Erode2);
        namedWindow( "Dialate", CV_WINDOW_AUTOSIZE );
        imshow("Dialate", Dialate2);
        //namedWindow( "Threshold", CV_WINDOW_AUTOSIZE );
        //imshow("Threshold", dst2);
        namedWindow( "Result", CV_WINDOW_AUTOSIZE );
        imshow("Result", src2);
        namedWindow( "Contour", CV_WINDOW_AUTOSIZE );
        imshow("Contour", drawing2);

        if(cvWaitKey( 15 ) == 27 ) break;
    }
  /// Wait until user finishes program
  /*while(true)
  {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 )
      { break; }
   }*/
}

int findBiggestContour( vector<vector<Point> > contours )
{
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;

    for( int i = 0; i < contours.size(); i++ )
    {
        if( contours[i].size() > sizeOfBiggestContour )
        {
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}
