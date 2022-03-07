#ifndef SRC_OBJECT_DETECTION_HPP_
#define SRC_OBJECT_DETECTION_HPP_

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

#include "object_detection_data.hpp"
#include "write_data.hpp"
#include "read_data.hpp"


void obj_detect(object_detection_data *frame_data_in, config_params config_params_values);


#endif
