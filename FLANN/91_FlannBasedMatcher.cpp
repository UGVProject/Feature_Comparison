//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV2���鱾����ʾ������91
//		����������ʹ��FLANN����������ƥ��
//		�����������ò���ϵͳ�� Windows 7 64bit
//		������������IDE�汾��Visual Studio 2010
//		������������OpenCV�汾��	2.4.9
//		2014��06�� Created by @ǳī_ë����
//		2014��11�� Revised by @ǳī_ë����
//------------------------------------------------------------------------------------------------



//---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
//		����������������ʹ�õ�ͷ�ļ��������ռ�
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


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//      ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );


//-----------------------------------��main( )������--------------------------------------------
//   ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	//��0���ı�console������ɫ
	system("color 4F");

	//��1������ԴͼƬ
	Mat img_1 = imread("/home/chris/projects/ImageRectify/build/rectified/image5_1.jpg" );
	Mat img_2 = imread( "/home/chris/projects/ImageRectify/build/rectified/image5_2.jpg" );
	if( !img_1.data || !img_2.data ) { printf("When reading images, something goes wrong~�� \n"); return false; }

	//��2������SURF��������Ĺؼ���
	int minHessian = 300;
	SURF detector( minHessian );
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector.detect( img_1, keypoints_1 );
	detector.detect( img_2, keypoints_2 );

	//��3������������������������
	SURF extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );

	//��4������FLANN�㷨ƥ������������
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0; double min_dist = 100; //100

	//��5�����ټ���ؼ���֮���������С����
	for( int i = 0; i < descriptors_1.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//���������Ϣ
	printf("> Max dist : %f \n", max_dist );
	printf("> Min dist : %f \n", min_dist );

	//��6�����·���������ƥ�������������С��2* min_dist�ģ���ʹ��radiusMatchͬ������
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_1.rows; i++ )
	{
		if( matches[i].distance < 2*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	//��7�����Ƴ�����������ƥ���
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//��8��������ƥ�����Ϣ
	for( int i = 0; i < good_matches.size(); i++ )
	{ printf( ">Matching points [%d] Feature1: %d  -- Feature2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

	//��9����ʾЧ��ͼ
	imshow( "Results", img_matches );

	//��������˳�����
	waitKey(0);
	return 0;
}