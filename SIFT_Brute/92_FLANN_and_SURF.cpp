//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV2���鱾����ʾ������92
//		����������FLANN���SURF���йؼ����������ƥ��
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
	//��0���ı�console������ɫ
	//system("color 6F");

	//void ShowHelpText();

	//��1������ͼ����ʾ��ת��Ϊ�Ҷ�ͼ

	Mat LeftImage = imread(img1_filename);
	Mat RightImage = imread(img2_filename);

	//��2�����Surf�ؼ��㡢��ȡѵ��ͼ��������
	vector<KeyPoint> train_keyPoint;
	Mat trainDescriptor;
	SurfFeatureDetector featureDetector(80);
	featureDetector.detect(LeftImage, train_keyPoint);
	SurfDescriptorExtractor featureExtractor;
	featureExtractor.compute(LeftImage, train_keyPoint, trainDescriptor);

	//��3����������FLANN��������ƥ�����
	FlannBasedMatcher matcher;
	vector<Mat> train_desc_collection(1, trainDescriptor);
	matcher.add(train_desc_collection);
	matcher.train();

	unsigned int frameCount = 0;//֡��

	//<1>��������
	int64 time0 = getTickCount();

	//<3>���S�ؼ��㡢��ȡ����ͼ��������
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
	cout << "FPS��" << getTickFrequency() / (getTickCount() - time0) << endl;
	cout << "match number = " << goodMatches.size() << endl;
	waitKey(0);

	return 0;
}
