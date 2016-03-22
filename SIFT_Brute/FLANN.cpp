
//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV2版书本配套示例程序95
//		程序描述：ORB的关键点和描述符的提取，使用FLANN-LSH进行匹配
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



//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	printf("\n\n\t\t\tThis is an ORB feature descriptor test program\n");
	printf("\n\n\t\t\t   The Opencv Version is: " CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	printf( "\n\n\tHelp: \n\n"
		"\t\tPress 'ESC' to quit\n" );
}


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
	//【0】改变console字体颜色
	system("color 2F");

	//【0】显示帮助文字
	ShowHelpText();

	//【0】载入源图，显示并转化为灰度图
	Mat LeftImage = imread(img1_filename);
	Mat RightImage = imread(img2_filename);
	//Mat LeftImage = imread("/home/chris/projects/ImageRectify/build/rectified/image5_1.jpg");
	//Mat RightImage = imread("/home/chris/projects/ImageRectify/build/rectified/image5_2.jpg");
	//imshow("Original image",srcImage);
	//Mat grayImage;
	//cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	//int num = 100;
	//------------------检测SIFT特征点并在图像中提取物体的描述符----------------------

	//【1】参数定义
	OrbFeatureDetector featureDetector;
	vector<KeyPoint> keyPoints, captureKeyPoints;
	Mat descriptors, captureDescription;

	//【2】调用detect函数检测出特征关键点，保存在vector容器中
	featureDetector.detect(LeftImage, keyPoints);

	//【3】计算描述符（特征向量）
	OrbDescriptorExtractor featureExtractor;

	double time0 = static_cast<double>(getTickCount( ));//记录起始时间
	featureExtractor.compute(LeftImage, keyPoints, descriptors);

	//【4】基于FLANN的描述符对象匹配
	flann::Index flannIndex(descriptors, flann::LshIndexParams(25, 20, 2), cvflann::FLANN_DIST_HAMMING);
	//flann::Index flannIndex(descriptors, flann::AutotunedIndexParams(0.9, 0.01, 0, 0.1));
	unsigned int frameCount = 0;//帧数

	//【8】调用detect函数检测出特征关键点，保存在vector容器中
	featureDetector.detect(RightImage, captureKeyPoints);

	//【9】计算描述符
	featureExtractor.compute(RightImage, captureKeyPoints, captureDescription);

	//【10】匹配和测试描述符，获取两个最邻近的描述符
	Mat matchIndex(captureDescription.rows, 2, CV_32SC1), matchDistance(captureDescription.rows, 2, CV_32FC1);
	flannIndex.knnSearch(captureDescription, matchIndex, matchDistance, 2, flann::SearchParams());//调用K邻近算法

	//【11】根据劳氏算法（Lowe's algorithm）选出优秀的匹配
	vector<DMatch> goodMatches;
	for(int i = 0; i < matchDistance.rows; i++)
	{
		if(matchDistance.at<float>(i, 0) < 0.7 * matchDistance.at<float>(i, 1))
		{
			DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			goodMatches.push_back(dmatches);
		}
	}

	//【12】绘制并显示匹配窗口
	Mat resultImage;
	drawMatches( RightImage, captureKeyPoints, LeftImage, keyPoints, goodMatches, resultImage);
	imshow("Matching Window", resultImage);

	//【13】显示帧率
	cout << ">FPS= " << getTickFrequency() / (getTickCount() - time0) << endl;
	waitKey(0);
	//num ++;

	return 0;
}


