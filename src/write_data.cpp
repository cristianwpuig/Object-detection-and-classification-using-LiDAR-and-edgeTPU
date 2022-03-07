////////////////////////////////////////////////////////////////////////////
//   File        : write_data.cpp
//   Description : In this file, the functions that write the information of the detected objects is presented.
//                 The write_data_prev_frame function writes the information about each detected object. This information
//                 is used in the next frame in the tracking stage. The write_voxel_in_txt function writes the voxels information
//                 of each detected object in a txt file which is then used in the object classification stage.
//
//   Created By: cristianwpuig (https://github.com/cristianwpuig)
////////////////////////////////////////////////////////////////////////////
#include "write_data.hpp"


// This function writes to the csv used to store the object detection data of each frame
void write_data_prev_frame(object_detection_data *object_detection_outputs){
	ofstream fout;
	fout.open ("../c_algorithm_outputs/object_detection_outputs.csv");
	fout << object_detection_outputs->time_between_frames ;
	fout << ",";
	fout << object_detection_outputs->is_first_frame ;
	fout << ",";
	if (object_detection_outputs->bounding_box_x_prev_frame.size() != 0){
		for (unsigned int i=0; i < object_detection_outputs->bounding_box_x_prev_frame.size();i++){
			fout << object_detection_outputs->bounding_box_x_prev_frame[i];
			fout << ",";
		}
	}
	fout << "-1";
	fout << "\n";

	if (object_detection_outputs->bounding_box_x_prev_frame.size() != 0){
		for (unsigned int i=0; i < object_detection_outputs->bounding_box_x_prev_frame.size();i++){
			fout << object_detection_outputs->object_velocity[i];
			fout << ",";
		}
	}
	fout << "-1";
	fout << "\n";

	if (object_detection_outputs->bounding_box_x_prev_frame.size() != 0){
		for (unsigned int i=0; i < object_detection_outputs->bounding_box_x_prev_frame.size();i++){
			fout << object_detection_outputs->object_area[i];
			fout << ",";
		}
	}
	fout << "-1";
	fout << "\n";

	// We increase the frame_ID
	fout << object_detection_outputs->frame_ID + 1;
	fout << ",";

	fout << "-1";
	fout << "\n";

	fout << "# First line: clock(), is first frame (0: no, 1:yes), vector(x_coor en pixels)\n";
	fout << "# Second line: vector (object velocity in m/s) for each detected objetc\n";
	fout << "# Third line: vector (object area in m2) for each detected objetc\n";
	fout.close();
}

// This function saves the voxels. A txt file is generated for each voxel detected in each frame.
void write_voxel_in_txt(vector< vector< vector<int> > > Voxel_data, int voxel_grid_size, int max_points_voxels, int objectID){
	ofstream fout;
	float final_voxel;
	string file_name = "../c_algorithm_outputs/detected_objects_in_voxels/voxel_object_" + to_string(objectID) + ".txt";
	fout.open (file_name);
	for (int i=0; i<voxel_grid_size;i++){
		for (int j=0; j<voxel_grid_size;j++){
			for (int k=0; k<voxel_grid_size;k++){
				// If the number is 0 we put it directly
				if (Voxel_data[k][j][i] == 0){
					fout << Voxel_data[k][j][i];
				}
				// If the number is !=0 the number is normalized depending on the voxel with the maximun number of points of this object
				else{
					final_voxel = floorf((float(Voxel_data[k][j][i])/float(max_points_voxels)) * 100) / 100;
					fout << final_voxel;
				}
				// Not to put the comma in the last element
				if (k < voxel_grid_size -1){
				fout << ",";
				}
			}
			fout << "\n";
		}
		fout << "Plane Z=";
		fout << i;
		fout << "\n";
	}
	fout << "EOD";
	fout << "\n";

	fout.close();
}
