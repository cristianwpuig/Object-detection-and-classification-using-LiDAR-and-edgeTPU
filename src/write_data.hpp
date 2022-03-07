#ifndef WRITE_DATA_H
#define WRITE_DATA_H

#include <fstream>
using namespace std;
#include "object_detection_data.hpp"


void write_data_prev_frame(object_detection_data *object_detection_outputs);
void write_voxel_in_txt(vector< vector< vector<int> > > Voxel_data, int voxel_grid_size, int max_points_voxels, int objectID);

#endif
