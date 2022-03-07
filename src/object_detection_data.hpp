
#ifndef OBJECT_DETECTION_DATA_H
#define OBJECT_DETECTION_DATA_H

#include <opencv2/video/tracking.hpp>
using namespace std;


struct object_detection_data
{
	vector<double> lidar_x;
	vector<double> lidar_y;
	vector<double> lidar_z;
	vector<double> lidar_lum;
    vector<int> bounding_box_x_prev_frame;
    double time_between_frames;
    bool is_first_frame;
    vector<float> object_velocity;
    vector<float> object_area;
    int frame_ID;

};


#endif
