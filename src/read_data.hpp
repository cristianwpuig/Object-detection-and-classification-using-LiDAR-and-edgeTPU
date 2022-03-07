#ifndef READ_DATA_H
#define READ_DATA_H

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "object_detection_data.hpp"

using namespace std;
using namespace cv;

struct config_params {
	bool PLANEXZtrue_XYfalse = false;
	bool IS_FIRST_FRAME_TEST = false;
	bool PLOT_IMAGES = true;
	bool PRINT_DEBUG_INFO = true;
	bool RESET_FRAME_ID = true;
	string POINT_CLOUD_DATASET_DIR;
	string DATASET_DATE_TIME;
	int STARTING_FRAME;
};

void read_outputs_previous_frame(object_detection_data *read_prev_frame);
void read_lidar_data(int frame_id, object_detection_data *frame_data_main, config_params config_params_values );
string getName(int num_frame, string CSVdir, config_params config_params_values);
config_params read_config_file();


#endif
