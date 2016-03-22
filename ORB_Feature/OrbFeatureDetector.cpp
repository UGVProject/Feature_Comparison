
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
	printf("\n\n\t\t\t   The Opencv Version is" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	printf( "\n\n\tHelp: \n\n"
		"\t\tPress 'ESC' to quit\n" );
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(	)
{
	//��0���ı�console������ɫ
	system("color 2F");

	//��0����ʾ��������
	ShowHelpText();

	//��0������Դͼ����ʾ��ת��Ϊ�Ҷ�ͼ
	Mat srcImage = imread("../2.jpg");
	imshow("Original image",srcImage);
	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);

	//------------------���SIFT�����㲢��ͼ������ȡ�����������----------------------

	//��1����������
	OrbFeatureDetector featureDetector;
	vector<KeyPoint> keyPoints;
	Mat descriptors;

	//��2������detect�������������ؼ��㣬������vector������
	featureDetector.detect(grayImage, keyPoints);

	//��3������������������������
	OrbDescriptorExtractor featureExtractor;
	featureExtractor.compute(grayImage, keyPoints, descriptors);

	//��4������FLANN������������ƥ��
	flann::Index flannIndex(descriptors, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

	//��5����ʼ����Ƶ�ɼ�����
	VideoCapture cap(0);

	unsigned int frameCount = 0;//֡��

	//��6����ѯ��ֱ������ESC���˳�ѭ��
	while(1)
	{
		double time0 = static_cast<double>(getTickCount( ));//��¼��ʼʱ��
		Mat  captureImage, captureImage_gray;//��������Mat������������Ƶ�ɼ�
		cap >>  captureImage;//�ɼ���Ƶ֡
		if( captureImage.empty())//�ɼ�Ϊ�յĴ���
			continue;

		//ת��ͼ�񵽻Ҷ�
		cvtColor( captureImage, captureImage_gray, CV_BGR2GRAY);//�ɼ�����Ƶ֡ת��Ϊ�Ҷ�ͼ

		//��7�����SIFT�ؼ��㲢��ȡ����ͼ���е�������
		vector<KeyPoint> captureKeyPoints;
		Mat captureDescription;

		//��8������detect�������������ؼ��㣬������vector������
		featureDetector.detect(captureImage_gray, captureKeyPoints);

		//��9������������
		featureExtractor.compute(captureImage_gray, captureKeyPoints, captureDescription);

		//��10��ƥ��Ͳ�������������ȡ�������ڽ���������
		Mat matchIndex(captureDescription.rows, 2, CV_32SC1), matchDistance(captureDescription.rows, 2, CV_32FC1);
		flannIndex.knnSearch(captureDescription, matchIndex, matchDistance, 2, flann::SearchParams());//����K�ڽ��㷨

		//��11�����������㷨��Lowe's algorithm��ѡ�������ƥ��
		vector<DMatch> goodMatches;
		for(int i = 0; i < matchDistance.rows; i++)
		{
			if(matchDistance.at<float>(i, 0) < 0.6 * matchDistance.at<float>(i, 1))
			{
				DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
				goodMatches.push_back(dmatches);
			}
		}

		//��12�����Ʋ���ʾƥ�䴰��
		Mat resultImage;
		drawMatches( captureImage, captureKeyPoints, srcImage, keyPoints, goodMatches, resultImage);
		imshow("ƥ�䴰��", resultImage);

		//��13����ʾ֡��
		cout << ">FPS= " << getTickFrequency() / (getTickCount() - time0) << endl;

		//����ESC����������˳�
		if(char(waitKey(1)) == 27) break;
	}

	return 0;
}

