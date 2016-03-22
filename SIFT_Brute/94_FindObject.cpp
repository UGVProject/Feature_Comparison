
//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV2���鱾����ʾ������94
//		����������ʹ�ö�ά������(Features2D)�͵�ӳ��(Homography)Ѱ����֪����
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
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//		������ȫ�ֺ���������
//-----------------------------------------------------------------------------------------------
static void ShowHelpText( );//�����������

//-----------------------------------��main( )������--------------------------------------------
//		����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
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
	{ printf("Image read wrong�� \n"); return false; }

	//��2��ʹ��SURF���Ӽ��ؼ���
	int minHessian = 400;//SURF�㷨�е�hessian��ֵ
	SurfFeatureDetector detector( minHessian );//����һ��SurfFeatureDetector��SURF�� ������������
	vector<KeyPoint> keypoints_object, keypoints_scene;//vectorģ���࣬����������͵Ķ�̬����

	//��3������detect��������SURF�����ؼ��㣬������vector������
	detector.detect( srcImage1, keypoints_object );
	detector.detect( srcImage2, keypoints_scene );

	//��4������������������������
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( srcImage1, keypoints_object, descriptors_object );
	extractor.compute( srcImage2, keypoints_scene, descriptors_scene );

	//��5��ʹ��FLANNƥ�����ӽ���ƥ��
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;//��С�����������

	//��6��������ؼ���֮���������ֵ����Сֵ
	for( int i = 0; i < descriptors_object.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf(">Max dist : %f \n", max_dist );
	printf(">Min dist : %f \n", min_dist );

	//��7������ƥ�����С��3*min_dist�ĵ��
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ )
	{
		if( matches[i].distance < 3*min_dist )
		{
			good_matches.push_back( matches[i]);
		}
	}

	//���Ƴ�ƥ�䵽�Ĺؼ���
	Mat img_matches;
	drawMatches( srcImage1, keypoints_object, srcImage2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//���������ֲ�����
	vector<Point2f> obj;
	vector<Point2f> scene;

	//��ƥ��ɹ���ƥ����л�ȡ�ؼ���
	for( unsigned int i = 0; i < good_matches.size(); i++ )
	{
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );//����͸�ӱ任

	//�Ӵ���ͼƬ�л�ȡ�ǵ�
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( srcImage1.cols, 0 );
	obj_corners[2] = cvPoint( srcImage1.cols, srcImage1.rows ); obj_corners[3] = cvPoint( 0, srcImage1.rows );
	vector<Point2f> scene_corners(4);

	//����͸�ӱ任
	perspectiveTransform( obj_corners, scene_corners, H);

	//���Ƴ��ǵ�֮���ֱ��
	line( img_matches, scene_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar(255, 0, 123), 4 );
	line( img_matches, scene_corners[1] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
	line( img_matches, scene_corners[2] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );
	line( img_matches, scene_corners[3] + Point2f( static_cast<float>(srcImage1.cols), 0), scene_corners[0] + Point2f( static_cast<float>(srcImage1.cols), 0), Scalar( 255, 0, 123), 4 );

	//��ʾ���ս��
	imshow( "Result", img_matches );

	waitKey(0);
	return 0;
}