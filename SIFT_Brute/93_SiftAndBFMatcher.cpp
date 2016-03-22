//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV2版书本配套示例程序93
//		程序描述：SIFT配合暴力匹配进行关键点描述和提取
//		开发测试所用操作系统： Windows 7 64bit
//		开发测试所用IDE版本：Visual Studio 2010
//		开发测试所用OpenCV版本：	2.4.9
//		2014年06月 Created by @浅墨_毛星云
//		2014年11月 Revised by @浅墨_毛星云
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
using namespace cv;
using namespace std;

//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
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

	//<1>参数设置
	double time0 = static_cast<double>(getTickCount( ));//记录起始时间

	//【2】检测SIFT关键点、提取训练图像描述符
	vector<KeyPoint> train_keyPoint;
	Mat trainDescription;
	SiftFeatureDetector featureDetector;
	featureDetector.detect(LeftImage, train_keyPoint);
	SiftDescriptorExtractor featureExtractor;
	featureExtractor.compute(LeftImage, train_keyPoint, trainDescription);

	// 【3】进行基于描述符的暴力匹配
	BFMatcher matcher;
	vector<Mat> train_desc_collection(1, trainDescription);
	matcher.add(train_desc_collection);
	matcher.train();

	//【4】创建视频对象、定义帧率

	unsigned int frameCount = 0;//帧数

	//<3>检测SURF关键点、提取测试图像描述符
	vector<KeyPoint> test_keyPoint;
	Mat testDescriptor;
	featureDetector.detect(RightImage, test_keyPoint);
	featureExtractor.compute(RightImage, test_keyPoint, testDescriptor);

	//<4>匹配训练和测试描述符
	vector<vector<DMatch> > matches;
	matcher.knnMatch(testDescriptor, matches, 2);

	// <5>根据劳氏算法（Lowe's algorithm），得到优秀的匹配点
	vector<DMatch> goodMatches;
	for(unsigned int i = 0; i < matches.size(); i++)
	{
		if(matches[i][0].distance < 0.6 * matches[i][1].distance)
			goodMatches.push_back(matches[i][0]);
	}

	//<6>绘制匹配点并显示窗口
	Mat dstImage;
	drawMatches(RightImage, test_keyPoint, LeftImage, train_keyPoint, goodMatches, dstImage);
	imshow("Result", dstImage);

	//<7>输出帧率信息
	cout << "\t>FPS: " << getTickFrequency() / (getTickCount() - time0) << endl;
	cout << "match number = " << goodMatches.size() << endl;
	waitKey(0);

	return 0;
}
