//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV2���鱾����ʾ������93
//		����������SIFT��ϱ���ƥ����йؼ�����������ȡ
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

	//ShowHelpText();

	Mat LeftImage = imread(img1_filename);
	Mat RightImage = imread(img2_filename);

	//<1>��������
	double time0 = static_cast<double>(getTickCount( ));//��¼��ʼʱ��

	//��2�����SIFT�ؼ��㡢��ȡѵ��ͼ��������
	vector<KeyPoint> train_keyPoint;
	Mat trainDescription;
	SiftFeatureDetector featureDetector;
	featureDetector.detect(LeftImage, train_keyPoint);
	SiftDescriptorExtractor featureExtractor;
	featureExtractor.compute(LeftImage, train_keyPoint, trainDescription);

	// ��3�����л����������ı���ƥ��
	BFMatcher matcher;
	vector<Mat> train_desc_collection(1, trainDescription);
	matcher.add(train_desc_collection);
	matcher.train();

	//��4��������Ƶ���󡢶���֡��

	unsigned int frameCount = 0;//֡��

	//<3>���SURF�ؼ��㡢��ȡ����ͼ��������
	vector<KeyPoint> test_keyPoint;
	Mat testDescriptor;
	featureDetector.detect(RightImage, test_keyPoint);
	featureExtractor.compute(RightImage, test_keyPoint, testDescriptor);

	//<4>ƥ��ѵ���Ͳ���������
	vector<vector<DMatch> > matches;
	matcher.knnMatch(testDescriptor, matches, 2);

	// <5>���������㷨��Lowe's algorithm�����õ������ƥ���
	vector<DMatch> goodMatches;
	for(unsigned int i = 0; i < matches.size(); i++)
	{
		if(matches[i][0].distance < 0.6 * matches[i][1].distance)
			goodMatches.push_back(matches[i][0]);
	}

	//<6>����ƥ��㲢��ʾ����
	Mat dstImage;
	drawMatches(RightImage, test_keyPoint, LeftImage, train_keyPoint, goodMatches, dstImage);
	imshow("Result", dstImage);

	//<7>���֡����Ϣ
	cout << "\t>FPS: " << getTickFrequency() / (getTickCount() - time0) << endl;
	cout << "match number = " << goodMatches.size() << endl;
	waitKey(0);

	return 0;
}
