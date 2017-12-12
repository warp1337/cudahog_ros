// ROS includes.
#include <ros/ros.h>

#include <ros/time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cudahog_ros/CudaHogDetections.h>

#include <string.h>
#include <QImage>
#include <QPainter>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cudaHOG.h>

#include "Matrix.h"
#include "Vector.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace cudahog_ros;

cudaHOG::cudaHOGManager *hog;
ros::Publisher pub_message;
image_transport::Publisher pub_result_image;

double score_thresh;

void render_bbox_2D(CudaHogDetections& detections, cv::Mat& image)
{
    for(int i = 0; i < detections.pos_x.size(); i++){
        int x =(int) detections.pos_x[i];
        int y =(int) detections.pos_y[i];
        int w =(int) detections.width[i];
        int h =(int) detections.height[i];
        float score = detections.score[i];
        cv::Point pt_1;
        pt_1.x = x;
        pt_1.y = y;
        cv::Point pt_2;
        pt_2.x = x+w;
        pt_2.y = y+h;
        cv::rectangle(image, pt_1, pt_2, cv::Scalar(255, 255, 255), 2, 8, 0);
    }
}

void imageCallback(const Image::ConstPtr &msg)
{
    std::vector<cudaHOG::Detection> detHog;

    // unsigned char image, this is required for libcudahog, unfortunately.
    QImage image_rgb(&msg->data[0], msg->width, msg->height, QImage::Format_RGB888);
    int returnPrepare = hog->prepare_image(image_rgb.convertToFormat(QImage::Format_ARGB32).bits(), (short unsigned int)msg->width, (short unsigned int)msg->height);

    if(returnPrepare)
    {
        ROS_ERROR(">>> Error while preparing the image for cudahog");
        return;
    }

    hog->test_image(detHog);
    hog->release_image();

    int w = 64, h = 128;

    CudaHogDetections detections;

    detections.header = msg->header;
    for(unsigned int i=0;i<detHog.size();i++)
    {
        float score = detHog[i].score;
        float scale = detHog[i].scale;
        float width = (w - 32.0f)*scale;
        float height = (h - 32.0f)*scale;
        float x = (detHog[i].x + 16.0f*scale);
        float y = (detHog[i].y + 16.0f*scale);
        if(detHog[i].score >= score_thresh) {
            detections.scale.push_back(scale);
            detections.score.push_back(score);
            detections.pos_x.push_back(x);
            detections.pos_y.push_back(y);
            detections.width.push_back(width);
            detections.height.push_back(height);
        }
    }

    if(pub_result_image.getNumSubscribers()) {
        ROS_DEBUG(">>> Got result, publishing image");
        if(detections.pos_x.size() > 0) {
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                render_bbox_2D(detections, cv_ptr->image);
                pub_result_image.publish(cv_ptr->toImageMsg());
            }
            catch (cv_bridge::Exception &e) {
                ROS_ERROR(">>> CV_BRIDGE exception: %s", e.what());
                return;
            }
        }
    }

    pub_message.publish(detections);
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber &sub_msg,
                     ros::NodeHandle &n,
                     string img_topic,
                     image_transport::SubscriberFilter &sub_col,
                     image_transport::ImageTransport &it){
    if(!pub_message.getNumSubscribers() && !pub_result_image.getNumSubscribers()) {
        ROS_INFO(">>> No subscribers. Unsubscribing.");
        sub_msg.shutdown();
        sub_col.unsubscribe();
    } else {
        ROS_INFO(">>> New subscribers. Subscribing.");
        sub_msg = n.subscribe(img_topic.c_str(), 1, &imageCallback);
        sub_col.subscribe(it,sub_col.getTopic().c_str(),1);
    }
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "cudahog");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    string camera_ns;
    string pub_topic;
    string pub_image_topic;
    string conf;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("model", conf, string(""));
    private_node_handle_.param("camera_namespace", camera_ns, string("/xtion/rgb"));
    private_node_handle_.param("score_thresh", score_thresh, 0.7);

    string image_color = camera_ns + "/image_raw";

    // Initialise cudaHOG
    if(strcmp(conf.c_str(),"") == 0) {
        ROS_ERROR(">>> No model path specified.");
        ROS_ERROR(">>> Run with: rosrun ... _model:=/path/to/model");
        exit(0);
    }

    hog = new  cudaHOG::cudaHOGManager();
    hog->read_params_file(conf);
    hog->load_svm_models();

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_message;
    image_transport::SubscriberFilter subscriber_color;
    subscriber_color.subscribe(it, image_color.c_str(), 1);
    subscriber_color.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(sub_message),
                                                       boost::ref(n),
                                                       image_color,
                                                       boost::ref(subscriber_color),
                                                       boost::ref(it));

    image_transport::SubscriberStatusCallback image_cb = boost::bind(&connectCallback,
                                                                   boost::ref(sub_message),
                                                                   boost::ref(n),
                                                                   image_color,
                                                                   boost::ref(subscriber_color),
                                                                   boost::ref(it));

    sub_message = n.subscribe(image_color.c_str(), 1, &imageCallback);

    // Create publishers
    private_node_handle_.param("detections", pub_topic, string("/cudahog/detections"));
    pub_message = n.advertise<CudaHogDetections>(pub_topic.c_str(), 2, con_cb, con_cb);

    private_node_handle_.param("result_image", pub_image_topic, string("/cudahog/image"));
    pub_result_image = it.advertise(pub_image_topic.c_str(), 2, image_cb, image_cb);

    ros::spin();

    return 0;
}

