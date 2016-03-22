
//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV2���鱾����ʾ������95
//		����������ORB�Ĺؼ��������������ȡ��ʹ��FLANN-LSH����ƥ��
//		�����������ò���ϵͳ�� Windows 7 64bit
//		������������IDE�汾��Visual Studio 2010
//		������������OpenCV�汾��	2.4.9
//		2014��06�� Created by @ǳī_ë����
//		2014��11�� Revised by @ǳī_ë����
//------------------------------------------------------------------------------------------------



//---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
//		����������������ʹ�õ�ͷ�ļ��������ռ�
//------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
using namespace cv;
using namespace std;



//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	printf("\n\n\t\t\tThis is an ORB feature descriptor test program\n");
	printf("\n\n\t\t\t   The Opencv Version is: " CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	printf( "\n\n\tHelp: \n\n"
		"\t\tPress 'ESC' to quit\n" );
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
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
	//��0���ı�console������ɫ
	system("color 2F");

	//��0����ʾ��������
	ShowHelpText();

	//��0������Դͼ����ʾ��ת��Ϊ�Ҷ�ͼ
	Mat LeftImage = imread(img1_filename);
	Mat RightImage = imread(img2_filename);
	//Mat LeftImage = imread("/home/chris/projects/ImageRectify/build/rectified/image5_1.jpg");
	//Mat RightImage = imread("/home/chris/projects/ImageRectify/build/rectified/image5_2.jpg");
	//imshow("Original image",srcImage);
	//Mat grayImage;
	//cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	//int num = 100;
	//------------------���SIFT�����㲢��ͼ������ȡ�����������----------------------

	//��1����������
	OrbFeatureDetector featureDetector;
	vector<KeyPoint> keyPoints, captureKeyPoints;
	Mat descriptors, captureDescription;

	//��2������detect�������������ؼ��㣬������vector������
	featureDetector.detect(LeftImage, keyPoints);

	//��3������������������������
	OrbDescriptorExtractor featureExtractor;

	double time0 = static_cast<double>(getTickCount( ));//��¼��ʼʱ��
	featureExtractor.compute(LeftImage, keyPoints, descriptors);

	//��4������FLANN������������ƥ��
	flann::Index flannIndex(descriptors, flann::LshIndexParams(25, 20, 2), cvflann::FLANN_DIST_HAMMING);
	//flann::Index flannIndex(descriptors, flann::AutotunedIndexParams(0.9, 0.01, 0, 0.1));
	unsigned int frameCount = 0;//֡��

	//��8������detect�������������ؼ��㣬������vector������
	featureDetector.detect(RightImage, captureKeyPoints);

	//��9������������
	featureExtractor.compute(RightImage, captureKeyPoints, captureDescription);

	//��10��ƥ��Ͳ�������������ȡ�������ڽ���������
	Mat matchIndex(captureDescription.rows, 2, CV_32SC1), matchDistance(captureDescription.rows, 2, CV_32FC1);
	flannIndex.knnSearch(captureDescription, matchIndex, matchDistance, 2, flann::SearchParams());//����K�ڽ��㷨

	//��11�����������㷨��Lowe's algorithm��ѡ�������ƥ��
	vector<DMatch> goodMatches;
	for(int i = 0; i < matchDistance.rows; i++)
	{
		if(matchDistance.at<float>(i, 0) < 0.7 * matchDistance.at<float>(i, 1))
		{
			DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			goodMatches.push_back(dmatches);
		}
	}

	//��12�����Ʋ���ʾƥ�䴰��
	Mat resultImage;
	drawMatches( RightImage, captureKeyPoints, LeftImage, keyPoints, goodMatches, resultImage);
	imshow("Matching Window", resultImage);

	//��13����ʾ֡��
	cout << ">FPS= " << getTickFrequency() / (getTickCount() - time0) << endl;
	waitKey(0);
	//num ++;

	return 0;
}


