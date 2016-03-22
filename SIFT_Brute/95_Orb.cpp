//--------------------------------------【程序说明】-------------------------------------------
//      程序说明：《OpenCV3编程入门》OpenCV2版书本配套示例程序92
//      程序描述：FLANN结合SURF进行关键点的描述和匹配
//      开发测试所用操作系统： Windows 7 64bit
//      开发测试所用IDE版本：Visual Studio 2010
//      开发测试所用OpenCV版本： 2.4.9
//      2014年06月 Created by @浅墨_毛星云
//      2014年11月 Revised by @浅墨_毛星云
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//      描述：包含程序所使用的头文件和命名空间
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
    //【1】载入图像、显示并转化为灰度图

    Mat LeftImage = imread(img1_filename);
    Mat RightImage = imread(img2_filename);

    //<1>参数设置
    int64 time0 = getTickCount();

    //【2】检测Orb关键点、提取训练图像描述符
    OrbFeatureDetector featureDetector;
    vector<KeyPoint> keyPoints, captureKeyPoints;
    Mat descriptors, captureDescription;
    //【2】调用detect函数检测出特征关键点，保存在vector容器中
    featureDetector.detect(LeftImage, keyPoints);
    OrbDescriptorExtractor featureExtractor;
    featureExtractor.compute(LeftImage, keyPoints, descriptors);

    //【3】创建基于FLANN的描述符匹配对象
    BFMatcher matcher(NORM_HAMMING2);
    //matcher= new BFMatcher<cv::HammingLUT>;

    unsigned int frameCount = 0;//帧数

    //<3>检测S关键点、提取测试图像描述符
    featureDetector.detect(RightImage, captureKeyPoints);
    featureExtractor.compute(RightImage, captureKeyPoints, captureDescription);

    //<4>匹配训练和测试描述符
    vector<vector<DMatch> > matches1;
    matcher.knnMatch(descriptors,captureDescription, matches1, 2);

    vector<vector<DMatch> > matches2;
    matcher.knnMatch(captureDescription, descriptors, matches2, 2);

    // <5>根据劳氏算法（Lowe's algorithm），得到优秀的匹配点
    vector<DMatch> goodMatches;
    for(unsigned int i = 0; i < matches1.size(); i++)
    {
        if(matches1[i][0].distance < 0.6 * matches1[i][1].distance)
            goodMatches.push_back(matches1[i][0]);
    }
    for(unsigned int i = 0; i < matches2.size(); i++)
    {
        if(matches2[i][0].distance < 0.6 * matches2[i][1].distance)
            goodMatches.push_back(matches2[i][0]);
    }

    //<6>绘制匹配点并显示窗口
    Mat dstImage;
    drawMatches(RightImage, captureDescription, LeftImage, descriptors, goodMatches, dstImage);
    imshow("Result", dstImage);

    //<7>输出帧率信息
    cout << "FPS：" << getTickFrequency() / (getTickCount() - time0) << endl;
    cout << "match number = " << goodMatches.size() << endl;
    waitKey(0);

    return 0;
}
