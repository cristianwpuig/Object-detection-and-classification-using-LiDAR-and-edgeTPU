////////////////////////////////////////////////////////////////////////////
//   File        : main.cpp
//   Description : This algorithm starts by reading a csv point cloud data generated using a Velodyne VLP-16 LiDAR. Then,
//				   all the objects inside a certain critical region are detected and tracked, the information is saved in
//				   ./c_algorithm_outputs/object_detection_outputs.csv. Finally each point cloud object is converted into
//                 voxel format and saved in .txt format in ./c_algorithm_outputs/detected_objects_in_voxels/. When the
//                 main function is called, one LiDAR frame is processed.
//
//   Created By: Cristian Wisultschew (https://github.com/cristianwpuig)
////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

#include "read_data.hpp"
#include "write_data.hpp"
#include "object_detection_data.hpp"
#include "object_detection.hpp"


config_params config_params_values;
object_detection_data frame_data_main;


int main()
{
	// Read config file located in ./python_scripts/config.txt
	config_params_values = read_config_file();
    //Initialize Object_detection_data struct
	frame_data_main.is_first_frame = true;
	frame_data_main.time_between_frames = 0;
	frame_data_main.object_velocity.push_back(11);
	frame_data_main.object_area.push_back(222);

	//Read data from previously frame sdaved in ./c_algorithm_outputs/object_detection_outputs.csv
	read_outputs_previous_frame(&frame_data_main);
	cout << "#####################" << endl;
	cout << "#### FRAME " << frame_data_main.frame_ID << " #######" << endl;
	cout << "#####################" << endl;
	// Read LiDAR point cloud from saved data
	read_lidar_data(frame_data_main.frame_ID, &frame_data_main, config_params_values);
	cout << "llegooou" << endl;
	// Perform object detection and tracking
	obj_detect(&frame_data_main, config_params_values);
	// Writing the results in ./c_algorithm_outputs/object_detection_outputs.csv
	write_data_prev_frame(&frame_data_main);

    return 0;
}

