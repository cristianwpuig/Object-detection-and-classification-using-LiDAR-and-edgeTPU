////////////////////////////////////////////////////////////////////////////
//   File        : object_detection.cpp
//   Description : In this file, the functions obj_detect perform the object detection and tracking. It also generates the voxels
//                 of each detected point cloud object.
//
//   Created By: Cristian Wisultschew (https://github.com/cristianwpuig)
////////////////////////////////////////////////////////////////////////////
#include "object_detection.hpp"


void obj_detect(object_detection_data *frame_data_in, config_params config_params_values)
{
	// Object detection parameters
	int pixel_w = 400;
	int pixel_h = 200;
	// Limit of the area of the box that is considered as detected object
	int box_limit_min = 1500;//2000;
	// At a pixel difference greater than ThresVal the pixels is painted in white.
	int thresVal = 30;//15
	// Parameters to detect contours
	int GaussBlur = 11;
	int border_type = 2;
	int med_blur = 15;
	// Parameters to dilates the contours
	int iterations = 18;//18
	// Display time of the plots
	int wait_k = 300;

//    // Limits XYZ of the ROI (Region Of Interest)
//    // Plane XY, DATASET_DATE_TIME = "2019-09-30-19-47-50"
//	double axes_limits[3][2] = {
//		{-2.1, 3},  // X axis range
//		{0.137, 2.5},  // Y axis range
//		{-0.7,1.2} // Z axis range
//	};

    // Limits XYZ of the ROI in ETSII and Austria datasets (plane XY)
	double axes_limits[3][2] = {
		{-9.4, 10},  // X axis range
		{1.6, 8.2},  // Y axis range
		{-0.9, 3.5} // Z axis range
	};

    // Object detection variables
	Mat firstFrame,frameDelta, thresh, debug_out, original_debug, out_1, aux_input;
	vector<vector<Point> > cnts, aux_cnts;
	Mat image(pixel_h, pixel_w, CV_8UC1, Scalar(0));
	float correctorX;
	float correctorZ;
	if (config_params_values.PLANEXZtrue_XYfalse == true){
		correctorX = (abs(axes_limits[0][0]) + abs(axes_limits[0][1]))/pixel_w;
		correctorZ = (abs(axes_limits[2][0]) + abs(axes_limits[2][1]))/pixel_h;
	}
	else{
		correctorX = (abs(axes_limits[0][0]) + abs(axes_limits[0][1]))/pixel_w;
		correctorZ = (abs(axes_limits[1][0]) + abs(axes_limits[1][1]))/pixel_h;
	}

	// Tracking variables
	int cnt_box = 0;
	int cnt_box_prev  = frame_data_in->bounding_box_x_prev_frame.size();
	vector<int> ID_box;
	int ID_min;
	double time_vel = 0;
	vector<int> tr_coord_x;
	vector<int> tr_coord_x_prev;
	vector<float> mov_x;
	vector<float> vel;

	// Voxel generation variables
	int voxel_grid_size = 16;
	vector<float> object_X;
	vector<float> object_Y;
	vector<float> object_Z;

	// LiDAR points variables
	double * dat_x;
	double * dat_y;
	double * dat_z;
	double * dat_lum;
	dat_x = new double [ frame_data_in->lidar_x.size()];
	dat_y = new double[ frame_data_in->lidar_x.size()];
	dat_z = new double [ frame_data_in->lidar_x.size()];
	dat_lum = new double[ frame_data_in->lidar_x.size()];
	for (int i = 0; i< frame_data_in->lidar_x.size(); i++){
		dat_x[i] = frame_data_in->lidar_x[i];
		dat_y[i] = frame_data_in->lidar_y[i];
		dat_z[i] = frame_data_in->lidar_z[i];
		dat_lum[i] = frame_data_in->lidar_lum[i];
	}
	int cnt = 0;
	// Filteres LiDAR points inside the ROI
	vector<float> ROI_data_x;
	vector<float> ROI_data_y;
	vector<float> ROI_data_z;
	vector<float> fdata_x;
	vector<float> fdata_z;
	vector<uchar> fdata_lum;

	// The LiDAR points inside the ROI are filtered. Note that if we are in the XY plane the z with the y are inverted.
	for (int i = 0; i<frame_data_in->lidar_x.size(); i++){
		if ((dat_x[i] < axes_limits[0][1]) and (dat_x[i] > axes_limits[0][0])){
			if ((dat_y[i] < axes_limits[1][1]) and (dat_y[i] > axes_limits[1][0])){
				if ((dat_z[i] < axes_limits[2][1]) and (dat_z[i] > axes_limits[2][0])){
					ROI_data_x.push_back(dat_x[i]);
					// Here we select the XY or YZ plane. Note that the points of y or z are stored in fdata_z.
					if (config_params_values.PLANEXZtrue_XYfalse==true){
						ROI_data_z.push_back(dat_z[i]);
						ROI_data_y.push_back(dat_y[i]);
					}
					else{
						ROI_data_z.push_back(dat_y[i]);
						ROI_data_y.push_back(dat_z[i]);
					}
					fdata_lum.push_back(dat_lum[i]);
					cnt++;

				}
			}
		}
	}
	if (config_params_values.PRINT_DEBUG_INFO)
		cout << "points after remove : " << cnt << " of a total input_x.size(): "<< frame_data_in->lidar_x.size() << endl;
	// We convert from meters to pixels using ROI limits
	for (int i = 0; i<cnt; i++){
	  fdata_x.push_back(((ROI_data_x [i]) - axes_limits[0][0])/correctorX);
	  fdata_z.push_back(((ROI_data_z [i]) - axes_limits[2][0])/correctorZ);
	  fdata_z [i] = double(pixel_h) - fdata_z [i];// put y=0 at the bottom instead of at the top (default)
	  fdata_lum[i] = fdata_lum[i] * 2.5;// we go from 0-100 of lum to 0-255 of pixel
	  // For security reasons, to avoid writing where it should not be written, it gives error if it is written in a pixel that does not exist.
	  if ((fdata_z[i] ) > pixel_h)
	    	fdata_z[i] = pixel_h - 1;
	  if ((fdata_z[i]) < 0)
			fdata_z[i] = 1;
	  if ((fdata_x[i]) > pixel_w)
			fdata_x[i] = pixel_w - 1;
	  if ((fdata_x[i]) < 0)
			fdata_x[i] = 1;
	  // Generate image BE CAREFUL NOT TO LEAVE NUMBERS => pixel_w and pixel_h or <= 0 because very rare errors will appear.
	  image.at<uchar>((static_cast<unsigned int>(fdata_z[i])),(static_cast<unsigned int>(fdata_x[i]))) = 255;
	}


	// In the first frame the background is saved if config param IS_FIRST_FRAME_TEST is true or if is_first_frame is set to 1 in in ./c_algorithm_outputs/object_detection_outputs.csv
	if (frame_data_in->is_first_frame == true or config_params_values.IS_FIRST_FRAME_TEST==true){
		firstFrame = image.clone();
		GaussianBlur(firstFrame, firstFrame, Size(GaussBlur, GaussBlur), border_type);
		imwrite("../c_algorithm_outputs/background.jpg", firstFrame); // For JPEG, it can be a quality ( CV_IMWRITE_JPEG_QUALITY ) from 0 to 100 (the higher is the better). Default value is 95.
		// Variables not to come back here
		frame_data_in->is_first_frame = false;
		config_params_values.IS_FIRST_FRAME_TEST = false;
		cout << "First frame saved" << endl;
	}

	if (frame_data_in->is_first_frame == false){
	aux_input = image.clone();
	GaussianBlur(image, image, Size(GaussBlur, GaussBlur), border_type);
	firstFrame = imread( "../c_algorithm_outputs/background.jpg", CV_LOAD_IMAGE_GRAYSCALE );//CV_LOAD_IMAGE_UNCHANGED);
	absdiff(image, firstFrame, frameDelta);
	threshold(frameDelta, thresh, thresVal, 255, THRESH_BINARY);
	dilate(thresh, thresh, Mat(), Point(-1,-1), iterations);
	Mat thres_cpy = thresh.clone();// Because findingcontours modifies the input image
	findContours(thres_cpy, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	drawContours(thres_cpy, cnts, -1, (255), 3);
	// Nothing is detected
	if (cnts.size() == 0){
		if (frame_data_in->bounding_box_x_prev_frame.size() != 0)
			frame_data_in->bounding_box_x_prev_frame.clear();
		if (frame_data_in->object_area.size() != 0)
			frame_data_in->object_area.clear();
		if (frame_data_in->object_velocity.size() != 0)
			frame_data_in->object_velocity.clear();
	}

	// Something is detected == there are boxes in cnts variable
	if (cnts.size() != 0){
		// aux_cnts is a copy of cnts but is destroyed by boundingRect()
		aux_cnts = cnts;
		// Count of the number of boxes in this frame == detected objects
		cnt_box = 0;
		// Save the X coordinates of the boxes of the previous frame.
		if (frame_data_in->bounding_box_x_prev_frame.size() != 0){
			tr_coord_x_prev.assign(frame_data_in->bounding_box_x_prev_frame.begin(), frame_data_in->bounding_box_x_prev_frame.end());
			frame_data_in->bounding_box_x_prev_frame.clear();
		}
		// We clean this array to save the X-coordinates of the boxes of this frame later on.
		if (frame_data_in->object_area.size() != 0)
			frame_data_in->object_area.clear();
		// Iterate through each of the boxes == detected objects
		for(unsigned int i = 0; i< cnts.size(); i++) {
			boundingRect(aux_cnts[i]);
			// If the box is bigger than a certain limit (this is produced because very small boxes are generated when there is nothing due to some single noise point)
			if(contourArea(cnts[i]) > box_limit_min) {
				// We save the limits of the box
				int cntx_max = 0;
				int cntz_max = 0;
				int cntx_min = 99999;
				int cntz_min = 99999;
				for (int i_1 = 0; i_1 < cnts[i].size(); i_1++){
					if (cnts[i][i_1].x > cntx_max)
						cntx_max = cnts[i][i_1].x;
					if (cnts[i][i_1].y > cntz_max)
						cntz_max = cnts[i][i_1].y;
					if (cnts[i][i_1].x < cntx_min)
						cntx_min = cnts[i][i_1].x;
					if (cnts[i][i_1].y < cntz_min)
						cntz_min = cnts[i][i_1].y;
				}
				// We draw the box (only in simulation)
				Point pt1(cntx_min,cntz_min);
				Point pt2(cntx_max,cntz_max);
				rectangle(image, pt1,pt2,128,2);
				// We store in array all the X-coordinates of the center of each box for the tracking stage.
				tr_coord_x.push_back((cntx_max + cntx_min)/2);
				frame_data_in->bounding_box_x_prev_frame.push_back(tr_coord_x[cnt_box]);

				if (config_params_values.PRINT_DEBUG_INFO)
					cout << "cntx_max, cntx_min, cntz_max, cntz_min: " << cntx_max << " " << cntx_min <<" " <<  cntz_max <<" " <<  cntz_min << endl;
				float alto = float(cntz_max - cntz_min);
				float ancho = float(cntx_max - cntx_min);
				alto = (alto * correctorZ) ;
				ancho = (ancho * correctorX);
				frame_data_in->object_area.push_back(ancho * alto);
				cnt_box++;

				// Save the points of the detected object to convert them to voxels Note that if the XY plane is used, the z and y are inverted.
				if (object_X.size() != 0){
					object_X.clear();
					object_Y.clear();
					object_Z.clear();
				}
				for (int ROI_points_cnt = 0; ROI_points_cnt<fdata_x.size(); ROI_points_cnt++){
					if (fdata_x[ROI_points_cnt] <= cntx_max and fdata_x[ROI_points_cnt] >= cntx_min){
						if (fdata_z[ROI_points_cnt] <= cntz_max and fdata_z[ROI_points_cnt] >= cntz_min){
							object_X.push_back(ROI_data_x[ROI_points_cnt]);
							if (config_params_values.PLANEXZtrue_XYfalse==true){
								object_Y.push_back(ROI_data_y[ROI_points_cnt]);
								object_Z.push_back(ROI_data_z[ROI_points_cnt]);
							}
							else{
								object_Y.push_back(ROI_data_z[ROI_points_cnt]);
								object_Z.push_back(ROI_data_y[ROI_points_cnt]);
							}
						}
					}
				}

				if (config_params_values.PRINT_DEBUG_INFO){
					cout << "fdata_x size: " << fdata_x.size() << endl;
					cout << "Object size: " << object_X.size() << endl;
				}
				// Generation of voxels
				// We calculate the limits of the object in meters
				float objx_max = -999;
				float objy_max = -999;
				float objz_max = -999;
				float objx_min = 999;
				float objy_min = 999;
				float objz_min = 999;
				for (int obj_points_cnt = 0; obj_points_cnt < object_X.size(); obj_points_cnt++){
					if (object_X[obj_points_cnt] > objx_max)
						objx_max = object_X[obj_points_cnt];
					if (object_Y[obj_points_cnt] > objy_max)
						objy_max = object_Y[obj_points_cnt];
					if (object_Z[obj_points_cnt] > objz_max)
						objz_max = object_Z[obj_points_cnt];
					if (object_X[obj_points_cnt] < objx_min)
						objx_min = object_X[obj_points_cnt];
					if (object_Y[obj_points_cnt] < objy_min)
						objy_min = object_Y[obj_points_cnt];
					if (object_Z[obj_points_cnt] < objz_min)
						objz_min = object_Z[obj_points_cnt];
				}

				if (config_params_values.PRINT_DEBUG_INFO){
					cout << "objx_max, objx_min: " << objx_max << ", " << objx_min << endl;
					cout << "objy_max, objy_min: " << objy_max << ", " << objy_min << endl;
					cout << "objz_max, objz_min: " << objz_max << ", " << objz_min << endl;
				}
				// We calculate the limits of each voxel and the width of the voxel in meters (since the points come in meters).
				// To do this we need to know the largest X Y or Z side to put as the side of the total voxel cube.
				float size_x, size_y, size_z, size_max;
				char size_max_axis;
				size_x = abs( objx_max- objx_min);
				size_y = abs( objy_max- objy_min);
				size_z = abs( objz_max- objz_min);
				for (int i_ = 0; i_ < 3; i_++){
					if (size_x >= size_y and size_x > size_z){
						size_max = size_x;
						size_max_axis = 'x';
					}
					if (size_y >= size_x and size_y > size_z){
						size_max = size_y;
						size_max_axis = 'y';
					}
					if (size_z >= size_x and size_z > size_y){
						size_max = size_z;
						size_max_axis = 'z';
					}
				}
				if (config_params_values.PRINT_DEBUG_INFO)
					cout << "size_x, size_y, size_z: " << size_x << ", " << size_y << ", " << size_z << " and size_max: " << size_max << endl;

				// Now the limits of each voxel are calculated in meters.
				// To do this we first calculate the midpoint of the object
				float middle_x, middle_y, middle_z;
				middle_x = objx_min + ((objx_max- objx_min)/2);
				middle_y = objy_min + ((objy_max- objy_min)/2);
				middle_z = objz_min + ((objz_max- objz_min)/2);

				vector<float> segment_length;
				float step;
				vector<float> segments_x, segments_y, segments_z;
				step = size_max/voxel_grid_size;
				// We calculate the segments in meters (taking into account the sign)
				for (int i_=0; i_<voxel_grid_size;i_++){
					segments_x.push_back(middle_x - (size_max/2) + step*i_);
					segments_y.push_back(middle_y - (size_max/2) + step*i_);
					segments_z.push_back(middle_z - (size_max/2) + step*i_);
					// We give a little margin to the boundary voxels in case there are just a few points on the boundaries.
					if (i_ == 0){
						segments_x[0] = segments_x[0] - 0.01;
						segments_y[0] = segments_y[0] - 0.01;
						segments_z[0] = segments_z[0] - 0.01;
					}
					if (i_ == voxel_grid_size - 1){
						segments_x[voxel_grid_size - 1] = segments_x[voxel_grid_size - 1] + 0.01;
						segments_y[voxel_grid_size - 1] = segments_y[voxel_grid_size - 1] + 0.01;
						segments_z[voxel_grid_size - 1] = segments_z[voxel_grid_size - 1] + 0.01;
					}

				}

				// We calculate in which voxel each point falls
				vector<int> Voxel_x(voxel_grid_size, 0), Voxel_y(voxel_grid_size, 0), Voxel_z(voxel_grid_size, 0);
				vector< vector< vector<int> > > Voxel(voxel_grid_size, vector< vector<int> >(voxel_grid_size , vector<int>(voxel_grid_size)));
				int cnt_pnts_voxel = 0;
				int max_points_voxels = 0;
				// For each object point
				for (int obj_pnts=0; obj_pnts < object_X.size(); obj_pnts++ ){
					// For each of the 16 voxels of the x-axis
					for (int cnt_voxl_size_x = 0; cnt_voxl_size_x < voxel_grid_size; cnt_voxl_size_x++){
						// We calculate in which voxel each point falls
						if(object_X[obj_pnts] >= segments_x[cnt_voxl_size_x] and object_X[obj_pnts] < segments_x[cnt_voxl_size_x + 1]){
							for (int cnt_voxl_size_y = 0; cnt_voxl_size_y < voxel_grid_size; cnt_voxl_size_y++){
								if(object_Y[obj_pnts] >= segments_y[cnt_voxl_size_y] and object_Y[obj_pnts] < segments_y[cnt_voxl_size_y + 1]){
									for (int cnt_voxl_size_z = 0; cnt_voxl_size_z < voxel_grid_size; cnt_voxl_size_z++){
										if(object_Z[obj_pnts] >= segments_z[cnt_voxl_size_z] and object_Z[obj_pnts] < segments_z[cnt_voxl_size_z + 1]){
											Voxel[cnt_voxl_size_x][cnt_voxl_size_y][cnt_voxl_size_z] += 1;
											cnt_pnts_voxel++;
											if (Voxel[cnt_voxl_size_x][cnt_voxl_size_y][cnt_voxl_size_z] > max_points_voxels){
												max_points_voxels = Voxel[cnt_voxl_size_x][cnt_voxl_size_y][cnt_voxl_size_z];
											}
										}
									}
								}
							}
						}
					}
				}

				// Save the Voxel in a txt file
				if (config_params_values.PRINT_DEBUG_INFO){
					cout << "puntos en voxel: "<< cnt_pnts_voxel << endl;
					cout << "max_points_voxels: " << max_points_voxels << endl;
				}
				write_voxel_in_txt(Voxel, voxel_grid_size, max_points_voxels, cnt_box - 1);
			}
		}
	}

	// Object tracking stage//
	// The time since the last time we were here is counted for the calculation of the velocity.
	time_vel = (double(clock() - frame_data_in->time_between_frames)/CLOCKS_PER_SEC);
	frame_data_in->time_between_frames = time_vel;
	if (config_params_values.PRINT_DEBUG_INFO)
		cout << "cnt_box: " << cnt_box << "  cnt_box_prev: " << cnt_box_prev << endl;
	// Distance that the box has moved in meters with respect to the previous frame
	vector<float> mov_x_min;
	// Nº boxes now == Nº boxes previous frame
	if ((cnt_box == cnt_box_prev) and (cnt_box != 0) ){
		if (config_params_values.PRINT_DEBUG_INFO)
			cout << "==== Cajas ahora == Cajas antes" << endl;
		// We loop through the boxes of the previous frame
		for (int i1=0 ; i1<cnt_box; i1++){
			float X_min_aux = 100000;
			mov_x.clear();
			// We loop through the boxes of this frame
			for (int i2=0 ; i2<cnt_box; i2++){
				// We save the distances between the previous box i1 and all current boxes i2.
				mov_x.push_back((float(abs(tr_coord_x[i1] - tr_coord_x_prev[i2])))*correctorX);
				if (config_params_values.PRINT_DEBUG_INFO)
					cout << "mov_x:  " << mov_x[i2] << " x coord: [" << i1 << "]: " << tr_coord_x[i1]  << " x coord prev[" << i2 << "]: " << tr_coord_x_prev[i2] << endl;
				// The smaller distance between the previous boxes (i1) and the current ones (i2) corresponds to the same box == smae box ID
				if (mov_x[i2] < X_min_aux){
					X_min_aux = mov_x[i2];
					ID_min = i2;
				}
			}
			// We store the dist X in meters and the ID of each of the boxes in order
			mov_x_min.push_back(mov_x[ID_min]);
			ID_box.push_back(ID_min);
			if (config_params_values.PRINT_DEBUG_INFO)
				cout << "ID_box[" << i1 <<"]: " << ID_min << endl;
		}


		// We calculate the speed with the dist X in meters and the time between frames
		frame_data_in->object_velocity.clear();
		for (int i=0 ; i<cnt_box; i++){
			vel.push_back(mov_x_min[i]/(time_vel));
			// We adjust the sign of the velocity depending on whether the object goes forwards or backwards
			if (tr_coord_x_prev[ID_box[i]] > tr_coord_x[i] )
				vel[i] = - vel[i];
			frame_data_in->object_velocity.push_back(vel[i]);
			if (config_params_values.PRINT_DEBUG_INFO){
				cout << "vel[" << i << "]: " << vel[i] << " in time: " << time_vel << endl;
				cout << "ID_box[i]: " << ID_box[i] <<" tr_coord_x_prev[ID_box[i]: " << tr_coord_x_prev[ID_box[i]] << " tr_coord_x[i]: " <<  tr_coord_x[i] << endl;
			}
		}
	}

	// Nº boxes now > Nº boxes previous frame
	// The same steps above are repeated
	if ((cnt_box > cnt_box_prev) and (cnt_box != 0) ){
		if (config_params_values.PRINT_DEBUG_INFO)
			cout << ">>>>>> Cajas ahora > Cajas antes" << endl;
		for (int i=0 ; i<cnt_box_prev ; i++){
			float X_min_aux = 100000;
			mov_x.clear();
			for (int i2=0 ; i2 < cnt_box; i2++){
				mov_x.push_back((float(abs(tr_coord_x[i2] - tr_coord_x_prev[i])))*correctorX);
				if (config_params_values.PRINT_DEBUG_INFO)
					cout << "mov_x:  " << mov_x[i2] << " x coord: [" << i2 << "]: " << tr_coord_x[i2]  << " x coord prev[" << i << "]: " << tr_coord_x_prev[i] << endl;
				if (mov_x[i2] < X_min_aux){
					X_min_aux = mov_x[i2];
					ID_min = i2;
				}
			}
			ID_box.push_back(ID_min);
			mov_x_min.push_back(mov_x[ID_min]);
			if (config_params_values.PRINT_DEBUG_INFO)
				cout << "ID_box: " << ID_box[i] << " mov_x_min: " << mov_x_min[i]  << endl;
		}

		frame_data_in->object_velocity.clear();
		for (int i=0 ; i< cnt_box_prev; i++){
			vel.push_back(mov_x_min[i]/(time_vel));
			if (tr_coord_x_prev[i] > tr_coord_x[ID_box[i]] )
				vel[i] = - vel[i];
			frame_data_in->object_velocity.push_back(vel[i]);
			if (config_params_values.PRINT_DEBUG_INFO){
				cout << "vel[" << i << "]: " << vel[i] << " in time: " << time_vel << endl;
				cout << "ID_box[i]: " << ID_box[i] <<" tr_coord_x_prev[ID_box[i]: " << tr_coord_x_prev[i] << " tr_coord_x[i]: " <<  tr_coord_x[ID_box[i]] << endl;
			}
		}
	}

	// Nº boxes now < Nº boxes previous frame
	// The same steps above are repeated
	if ((cnt_box < cnt_box_prev) and (cnt_box != 0) ){
		if (config_params_values.PRINT_DEBUG_INFO)
			cout << "<<<<<< Cajas ahora < Cajas antes" << endl;
		for (int i=0 ; i<cnt_box ; i++){
			float X_min_aux = 100000;
			mov_x.clear();
			for (int i2=0 ; i2 < cnt_box_prev; i2++){
				mov_x.push_back((float(abs(tr_coord_x[i] - tr_coord_x_prev[i2])))*correctorX);
				if (config_params_values.PRINT_DEBUG_INFO)
					cout << "mov_x:  " << mov_x[i2] << " x coord: [" << i << "]: " << tr_coord_x[i]  << " x coord prev[" << i2 << "]: " << tr_coord_x_prev[i2] << endl;
				if (mov_x[i2] < X_min_aux){
					X_min_aux = mov_x[i2];
					ID_min = i2;
				}
			}
			ID_box.push_back(ID_min);
			mov_x_min.push_back(mov_x[ID_min]);
			if (config_params_values.PRINT_DEBUG_INFO)
				cout << "ID_box: " << ID_box[i] << " mov_x[" << i << "]: " << mov_x_min[i] << endl;
		}

		frame_data_in->object_velocity.clear();
		for (int i=0 ; i< cnt_box; i++){
			vel.push_back(mov_x_min[i]/(time_vel));
			if (tr_coord_x_prev[ID_box[i]] > tr_coord_x[i] )
				vel[i] = - vel[i];
			frame_data_in->object_velocity.push_back(vel[i]);
			if (config_params_values.PRINT_DEBUG_INFO){
				cout << "vel[" << i << "]: " << vel[i] << " in time: " << time_vel << endl;
				cout << "ID_box[i]: " << ID_box[i] <<" tr_coord_x_prev[ID_box[i]: " << tr_coord_x_prev[ID_box[i]] << " tr_coord_x[i]: " <<  tr_coord_x[i] << endl;
			}
		}
	}

	// Plot the images (Only in debugging mode)
	if (config_params_values.PLOT_IMAGES == true) {
		Mat win_mat1, win_mat2, win_mat3;
		hconcat(image, thresh, win_mat1);
		hconcat(aux_input, frameDelta, win_mat3);
		vconcat(win_mat1, win_mat3, win_mat2);
		imshow( "1_Display Image", win_mat2 );
		waitKey(wait_k);
		destroyWindow("1_Display Image");
	}

	// STOP (Only in debugging mode)
	if (config_params_values.STOP_BETWEEN_FRAMES == true){
		cout << "System stopped, press enter to continue..." << endl;
		cin.get();
	}
	}// closing of is_first_frame == false
}
