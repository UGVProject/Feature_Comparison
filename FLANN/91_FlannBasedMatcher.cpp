//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV2版书本配套示例程序91
//		程序描述：使用FLANN进行特征点匹配
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
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;


//-----------------------------------【全局函数声明部分】--------------------------------------
//      描述：全局函数声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );


//-----------------------------------【main( )函数】--------------------------------------------
//   描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	//【0】改变console字体颜色
	system("color 4F");

	//【1】载入源图片
	Mat img_1 = imread("/home/chris/projects/ImageRectify/build/rectified/image5_1.jpg" );
	Mat img_2 = imread( "/home/chris/projects/ImageRectify/build/rectified/image5_2.jpg" );
	if( !img_1.data || !img_2.data ) { printf("When reading images, something goes wrong~！ \n"); return false; }

	//【2】利用SURF检测器检测的关键点
	int minHessian = 300;
	SURF detector( minHessian );
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector.detect( img_1, keypoints_1 );
	detector.detect( img_2, keypoints_2 );

	//【3】计算描述符（特征向量）
	SURF extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );

	//【4】采用FLANN算法匹配描述符向量
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0; double min_dist = 100; //100

	//【5】快速计算关键点之间的最大和最小距离
	for( int i = 0; i < descriptors_1.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//输出距离信息
	printf("> Max dist : %f \n", max_dist );
	printf("> Min dist : %f \n", min_dist );

	//【6】存下符合条件的匹配结果（即其距离小于2* min_dist的），使用radiusMatch同样可行
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_1.rows; i++ )
	{
		if( matches[i].distance < 2*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	//【7】绘制出符合条件的匹配点
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//【8】输出相关匹配点信息
	for( int i = 0; i < good_matches.size(); i++ )
	{ printf( ">Matching points [%d] Feature1: %d  -- Feature2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

	//【9】显示效果图
	imshow( "Results", img_matches );

	//按任意键退出程序
	waitKey(0);
	return 0;
}
