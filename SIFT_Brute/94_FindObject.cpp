
//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV2版书本配套示例程序94
//		程序描述：使用二维特征点(Features2D)和单映射(Homography)寻找已知物体
//		开发测试所用操作系统： Windows 7 64bit
//		开发测试所用IDE版本：Visual Studio 2010
//		开发测试所用OpenCV版本：	2.4.9
//		2014年06月 Created by @浅墨_毛星云
//		2014年11月 Revised by @浅墨_毛星云
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );//输出帮助文字

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	const char* img1_filename = 0;
    const char* img2_filename = 0;

    for( int i = 1; i < argc; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
    }
	Mat srcImage1 = imread(img1_filename);
	Mat srcImage2 = imread(img2_filename);
	if( !srcImage1.data || !srcImage2.data )
	{ printf("Image read wrong！ \n"); return false; }

	//【2】使用SURF算子检测关键点
	int minHessian = 400;//SURF算法中的hessian阈值
	SurfFeatureDetector detector( minHessian );//定义一个SurfFeatureDetector（SURF） 特征检测类对象
	vector<KeyPoint> keypoints_object, keypoints_scene;//vector模板类，存放任意类型的动态数组

	//【3】调用detect函数检测出SURF特征关键点，保存在vector容器中
	detector.detect( srcImage1, keypoints_object );
	detector.detect( srcImage2, keypoints_scene );

	//【4】计算描述符（特征向量）
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( srcImage1, keypoints_object, descriptors_object );
	extractor.compute( srcImage2, keypoints_scene, descriptors_scene );

	//【5】使用FLANN匹配算子进行匹配
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;//最小距离和最大距离

	//【6】计算出关键点之间距离的最大值和最小值
	for( int i = 0; i < descriptors_object.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf(">Max dist : %f \n", max_dist );
	printf(">Min dist : %f \n", min_dist );

	//【7】存下匹配距离小于3*min_dist的点对
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ )
	{
		if( matches[i].distance < 3*min_dist )
		{
			good_matches.push_back( matches[i]);
		}
	}

	//绘制出匹配到的关键点
	Mat img_matches;
	drawMatches( srcImage1, keypoints_object, srcImage2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//定义两个局部变量
	vector<Point2f> obj;
	vector<Point2f> scene;

	//从匹配成功的匹配对中获取关键点
	for( unsigned int i = 0; i < good_matches.size(); i++ )
	{
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );//计算透视变换

	//从待测图片中获取角点
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( srcImage1.cols, 0 );
	obj_corners[2] = cvPoint( srcImage1.cols, srcImage1.rows ); obj_corners[3] = cvPoint( 0, srcImage1.rows );
	vector<Point2f> scene_corners(4);

	//进行透视变换
	perspectiveTransform( obj_corners, scene_corners, H);

	//绘制出角点之间的直线
	line( img_matches, scene_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar(255, 0, 123), 4 );
	line( img_matches, scene_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
	line( img_matches, scene_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
	line( img_matches, scene_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );

	//显示最终结果
	imshow( "Result", img_matches );

	waitKey(0);
	return 0;
}
