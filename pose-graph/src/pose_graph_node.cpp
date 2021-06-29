#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <ctime>
#include "utility/tic_toc.h"
#include "LoopClosing.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"

#include <okvis_ros/SvinHealth.h>  // for svin_health publisher
#include "KFMatcher.h"

// Hunter
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;

map<int, KFMatcher*> kfMapper_;// Mapping between kf_index and KFMatcher*; to make KFcounter

queue<Eigen::Vector3d> svinOdomBuffer_;
int frame_index  = 0;
int sequence = 1;
LoopClosing posegraph;
BriefVocabulary* voc;
BriefDatabase db;


int SKIP_CNT = 0;
int skip_cnt = 0;

bool start_flag = 0;
double SKIP_DIS = 0;


int MIN_LOOP_NUM;
int FAST_RELOCALIZATION;


double p_fx, p_fy, p_cx, p_cy;  // projection_matrix

Eigen::Vector3d tic;
Eigen::Matrix3d qic;

ros::Publisher pubMatchedPoints;
ros::Publisher pubCamPoseVisual;
ros::Publisher pubKfOdom;


std::string BRIEF_PATTERN_FILE;

std::string SVIN_W_LOOP_PATH;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

void svinCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    Vector3d trans(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Quaterniond quat;
    quat.w() = msg->pose.pose.orientation.w;
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;

    trans = posegraph.w_r_svin * trans + posegraph.w_t_svin;
    quat = posegraph.w_r_svin *  quat;

    trans = posegraph.r_drift * trans + posegraph.t_drift;
    quat = posegraph.r_drift * quat;

    Vector3d svin_t_cam;
    Quaterniond svin_q_cam;
    svin_t_cam = trans + quat * tic;
    svin_q_cam = quat * qic;

	cameraposevisual.reset();
	cameraposevisual.add_pose(svin_t_cam, svin_q_cam);
	cameraposevisual.publish_by(pubCamPoseVisual, msg->header);


    svinOdomBuffer_.push(svin_t_cam);
    if (svinOdomBuffer_.size() > 10)
    {
        svinOdomBuffer_.pop();
    }

    visualization_msgs::Marker key_odometrys;
    key_odometrys.header = msg->header;
    key_odometrys.header.frame_id = "world";
    key_odometrys.ns = "key_odometrys";
    key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
    key_odometrys.action = visualization_msgs::Marker::ADD;
    key_odometrys.pose.orientation.w = 1.0;
    key_odometrys.lifetime = ros::Duration();

    //static int key_odometrys_id = 0;
    key_odometrys.id = 0; //key_odometrys_id++;
    key_odometrys.scale.x = 0.1;
    key_odometrys.scale.y = 0.1;
    key_odometrys.scale.z = 0.1;
    key_odometrys.color.r = 1.0;
    key_odometrys.color.a = 1.0;

    for (unsigned int i = 0; i < svinOdomBuffer_.size(); i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d svin_t;
        svin_t = svinOdomBuffer_.front();
        svinOdomBuffer_.pop();
        pose_marker.x = svin_t.x();
        pose_marker.y = svin_t.y();
        pose_marker.z = svin_t.z();
        key_odometrys.points.push_back(pose_marker);
        svinOdomBuffer_.push(svin_t);
    }
    pubKfOdom.publish(key_odometrys);
}

void processMeasurements(const sensor_msgs::ImageConstPtr& image_msg,
                         const sensor_msgs::PointCloudConstPtr& point_msg,
                         const nav_msgs::Odometry::ConstPtr& pose_msg,
                         const okvis_ros::SvinHealth::ConstPtr& health_msg=nullptr) {
            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                return;
            }
            else
            {
                skip_cnt = 0;
            }

            if (health_msg != nullptr and !health_msg->isTrackingOk) {
                return;  // Drop keyframes when tracking fails
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::KeyPoint> point_2d_uv;
                vector<Eigen::Vector3d> point_ids; // @Reloc: landmarkId, mfId, keypointIdx related to each point
                // For every KF, a map <observed_kf, weight> describing which other kfs how many MapPoints are common.
                map<KFMatcher*, int> KFcounter;

                int kf_index = -1;
                cv::Mat temp_image = image;
                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);



                    // @Reloc
                    Eigen::Vector3d p_ids;
                    p_ids(0) = point_msg->channels[i].values[0];   // landmarkId
                    p_ids(1) = point_msg->channels[i].values[1];  // poseId or MultiFrameId
                    p_ids(2) = point_msg->channels[i].values[2];  //keypointIdx
                    point_ids.push_back(p_ids);


                    cv::KeyPoint p_2d_uv;
                    double p_id;
                    kf_index = point_msg->channels[i].values[3];   // TODO Sharmin: this is redundant. This is same for the entire for loop.
                    p_2d_uv.pt.x = point_msg->channels[i].values[4];
                    p_2d_uv.pt.y = point_msg->channels[i].values[5];
                    p_2d_uv.size = point_msg->channels[i].values[6];
                    p_2d_uv.angle = point_msg->channels[i].values[7];
                    p_2d_uv.octave = point_msg->channels[i].values[8];
                    p_2d_uv.response = point_msg->channels[i].values[9];
                    p_2d_uv.class_id = point_msg->channels[i].values[10];

                    point_2d_uv.push_back(p_2d_uv);

                    //std::cout<<"CV Keypoint of size 8:"<< p_2d_uv.pt.x << " , "<< p_2d_uv.pt.y<< " size: "<< p_2d_uv.size<< " angle: "<<
                    //		p_2d_uv.angle << " octave: "<< p_2d_uv.octave<< " response: "<< p_2d_uv.response<< " class_id: "<<
					//		p_2d_uv.class_id << std::endl;



                    for (size_t sz = 11; sz < point_msg->channels[i].values.size(); sz++){
                    	int observed_kf_index = point_msg->channels[i].values[sz]; //kf_index where this point_3d has been observed
                    	if ( observed_kf_index == kf_index ){
                    		continue;
                    	}

                    	map<int, KFMatcher*>::iterator mkfit;
                    	mkfit = kfMapper_.find(observed_kf_index);
                    	if (mkfit == kfMapper_.end()){
                    		continue;
                    	}


                    	KFMatcher* observed_kf = kfMapper_.find(observed_kf_index)->second; //Keyframe where this point_3d has been observed
                    	KFcounter[observed_kf]++;
                    }


                }

                KFMatcher* keyframe = new KFMatcher(pose_msg->header.stamp.toSec(), point_ids, kf_index, T, R, image,
                                   point_3d, point_2d_uv, KFcounter, sequence, voc);

                kfMapper_.insert(std::make_pair(kf_index, keyframe));

					start_flag = 1;
					posegraph.addKFToPoseGraph(keyframe, 1);
                frame_index++;
                last_t = T;
            }
}

static std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    char s[100];
    std::strftime(s, sizeof(s), "%Y_%m_%d_%H_%M_%S", std::localtime(&now));
    return s;
}

void readParameters(ros::NodeHandle& nh)
{
    std::string config_file;
    nh.getParam("config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    double camera_visual_size = fsSettings["visualize_camera_size"];
    cameraposevisual.setScale(camera_visual_size);
    cameraposevisual.setLineWidth(camera_visual_size / 10.0);


    std::string IMAGE_TOPIC;

    // Read config file parameters
	double resize_factor = static_cast<double>(fsSettings["resizeFactor"]);;

	cv::FileNode fnode = fsSettings["projection_matrix"];
	p_fx = static_cast<double>(fnode["fx"]);
	p_fy = static_cast<double>(fnode["fy"]);
	p_cx = static_cast<double>(fnode["cx"]);
	p_cy = static_cast<double>(fnode["cy"]);

	if (resize_factor != 1.0){
	  p_fx = p_fx * resize_factor;
	  p_fy = p_fy * resize_factor;
	  p_cx = p_cx * resize_factor;
	  p_cy = p_cy * resize_factor;
	}
	cout << "projection_matrix: " << p_fx << " " <<
					  p_fy << " " << p_cx << " " <<p_cy<< endl;


	std::string pkg_path = ros::package::getPath("pose_graph");


	string vocabulary_file = pkg_path + "/Vocabulary/brief_k10L6.bin";
	cout << "vocabulary_file" << vocabulary_file << endl;

	// Loading vocabulary
	voc = new BriefVocabulary(vocabulary_file);
	db.setVocabulary(*voc, false, 0);
	posegraph.setBriefVocAndDB(voc, db);

	BRIEF_PATTERN_FILE = pkg_path + "/Vocabulary/brief_pattern.yml";

	MIN_LOOP_NUM = fsSettings["min_loop_num"];
	cout << "Num of matched keypoints for Loop Detection: " << MIN_LOOP_NUM << endl;

	FAST_RELOCALIZATION = fsSettings["fast_relocalization"];

	SVIN_W_LOOP_PATH = pkg_path + "/svin_results/svin_"+getTimeStr()+".txt";

	cout<<"SVIN Result path: "<<SVIN_W_LOOP_PATH<<endl;
	std::ofstream fout(SVIN_W_LOOP_PATH, std::ios::out);
	fout.close();
	fsSettings.release();
}

#define MAKE_SYNCHRONIZER_4(name, Policy, sub1, sub2, sub3, sub4, type4, callback, queue_size, connect) \
    typedef message_filters::sync_policies::Policy<sensor_msgs::Image, sensor_msgs::PointCloud, nav_msgs::Odometry, type4> name ## policy; \
    message_filters::Synchronizer<name ## policy> name(name ## policy(queue_size)); \
    if (connect) { \
        name.connectInput(sub1, sub2, sub3, sub4); \
        name.registerCallback(callback); \
    }
#define MAKE_SYNCHRONIZER_3(name, Policy, sub1, sub2, sub3, callback, queue_size, connect) \
    message_filters::NullFilter<message_filters::NullType> name ## M4; \
    MAKE_SYNCHRONIZER_4(name, Policy, sub1, sub2, sub3, name ## M4, message_filters::NullType, \
                        boost::bind(callback, _1, _2, _3, nullptr), queue_size, connect)

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_graph");
    ros::NodeHandle nh("~");
    posegraph.setPublishers(nh);

    // read parameters
    readParameters(nh);

    // Approx time
    bool approximate_sync = false;
    nh.getParam("approximate_sync", approximate_sync);

    // Optional connection to svin_health
    bool use_health = false;
    nh.getParam("use_health", use_health);

    // Subscribers
    ros::Subscriber subSVIN = nh.subscribe("/okvis_node/relocalization_odometry", 500, svinCallback);
    message_filters::Subscriber<sensor_msgs::Image> subKF(nh, "/okvis_node/keyframe_imageL", 500);
    message_filters::Subscriber<sensor_msgs::PointCloud> subPCL(nh, "/okvis_node/keyframe_points", 500);
    message_filters::Subscriber<nav_msgs::Odometry> subKFPose(nh, "/okvis_node/keyframe_pose", 500);

    // For UberEstimator
    //ros::Subscriber subPEPose = nh.subscribe("/PE_pose", 100, peCallback);  // Primitive Estimator topic
    message_filters::Subscriber<okvis_ros::SvinHealth> subSVINHealth;
    if (use_health) {
        subSVINHealth.subscribe(nh, "/okvis_node/svin_health", 500);
    }

    // Publishers
    pubCamPoseVisual = nh.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pubKfOdom = nh.advertise<visualization_msgs::Marker>("key_odometrys", 1000);
    pubMatchedPoints = nh.advertise<sensor_msgs::PointCloud>("match_points", 100); // to publish matched points after relocalization

    // Synchronizer
    MAKE_SYNCHRONIZER_3(exact_sync, ExactTime, subKF, subPCL, subKFPose,
                        processMeasurements, 100, !approximate_sync && !use_health);
    MAKE_SYNCHRONIZER_3(approx_sync, ApproximateTime, subKF, subPCL, subKFPose,
                        processMeasurements, 100, approximate_sync && !use_health);
    MAKE_SYNCHRONIZER_4(exact_sync_with_health, ExactTime, subKF, subPCL, subKFPose, subSVINHealth,
                        okvis_ros::SvinHealth, processMeasurements, 100, !approximate_sync && use_health);
    MAKE_SYNCHRONIZER_4(approx_sync_with_health, ApproximateTime, subKF, subPCL, subKFPose, subSVINHealth,
                        okvis_ros::SvinHealth, processMeasurements, 100, approximate_sync && use_health);

    ros::AsyncSpinner spinner(2);  // Keep two separate threads - svinCallback and processMeasurements
    spinner.start();
    ros::waitForShutdown();

    return 0;
}
