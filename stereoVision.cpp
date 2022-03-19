#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
using namespace std;
using namespace Eigen;


//文件路径
string left_file = "./left.png";
string right_file = "./right.png";
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);
int main(int argc, char **argv)
{
 //内参
double fx = 718.856, fy = 718.856, cx = 607.1982, cy = 185.2175;
//基线
double b = 0.573;
//读取图像
cv::Mat left = cv::imread(left_file, 0);
cv::Mat right = cv::imread(right_file, 0);
cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32); //神奇的参数
cv::Mat disparity_sgbm, disparity;
sgbm -> compute(left, right, disparity_sgbm);
disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

//生成点云
vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
//如果你的机器慢，请把后面的V++和u++改成v+=2, u+=2
for (int v = 0; v < left.rows; v++)
   for (int u = 0; u < left.cols; u++)
   {
     if (disparity.at<float>(v, u) <= 10.0 || disparity.at<float>(v, u) >= 96.0) continue;
     
     Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);  //前三维为xyz，第四维为颜色
     //根据双目模型计算point的位置
     double x = (u - cx) / fx;
     double y = (v - cy) / fy;
     double depth = fx * b / (disparity.at<float>(v, u));
     point[0] = x * depth;
     point[1] = y * depth;
     point[2] = depth;
     pointcloud.push_back(point);
   }

  cv::imshow("disparity", disparity / 96.0);
  cv::waitKey(0);
  //画出点云
  showPointCloud(pointcloud);
  return 0;

}
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    //glEnable(GL_DEPTH_TEST)启用了之后，OpenGL在绘制的时候就会检查，当前像素前面是否有别的像素，如果别的像素挡道了它，那它就不会绘制，也就是说，OpenGL就只绘制最前面的一层。
    //当我们需要绘制透明图片时，就需要关闭它glDisable(GL_DEPTH_TEST);
    //并且打开混合 glEnable(GL_BLEND);
    //源因子和目标因子是可以通过glBlendFunc函数来进行设置的。
    // glBlendFunc有两个参数，前者sfactor表示源因子，后者dfactor表示目标因子。
    //前者sfactor表示源颜色，后者dfactor表示目标颜色
    //GL_SRC_ALPHA：表示使用源颜色的alpha值来作为因子
    //GL_ONE_MINUS_SRC_ALPHA：表示用1.0减去源颜色的alpha值来作为因子（1-alpha)
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    //OpenGlRenderState 创建一个相机的观察视图，模拟相机
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );//ProjectionMatrix 前两个参数是相机的宽高，紧接着四个参数是相机的内参，最后两个是最近和最远视距
//ModelViewLookAt 前三个参数是相机的位置，紧接着三个是相机所看的视点的位置，最后三个参数是一个向量，表示相机的的朝向
    //在窗口创建交互式视图
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
 //SetBounds() 前四个参数是视图在视窗中的范围（下 上 左 右）
 //     SetHandle设置相机的视图句柄，需要用它来显示前面设置的 “相机” 所 “拍摄” 的内容
    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
