# Object detection, tracking and classification using LiDAR sensor and Coral edgeTPU to accelerate.

## What is This?
The system presentes provides object detection, tracking and classification of each object that pass trhoug a predefined ROI. The system input is a LiDAR point cloud. The object classification stage is accelerated with the Coral edgeTPU DL accelerator. The system is divided in two main parts. First, the object dtection and tracking stage is performed. This stage is encapsulated into a C++ function, which is called by a python script. In this stage the bounding boxes of each detected object are generated along with information about ID, location and speed of each object. In this stage the point cloud of each detected object is convertrd into voxels format since is the format admitted by the DNN that perform the object classification task. When the first stage end starts the second stage executed in python. In this stage, each object in voxels format is given as input to a DNN that runs into the Coral edgeTPU to provide the class of this object. As this system is developed for surveillance applications in street scenarios the classes that is capable of distinguish are: cars, pedestrians, trucks, and bicycles/motorbikes.

## Hardware requirements
* Desktop PC or SBC (Implemented and tested on Ubuntu 18.04 and Debian 10)
* LiDAR sensor (Tested with Velodyne VLP-16)
* DL accelerator (Tested with Coral USB edgeTPU)

## Software Requirements
Make sure you have installed all of the following prerequisites on your development machine:
* Python3 >= 3.6
* OpenCV == 3.4
* tflite-runtime == 2.5
* boost == 1.47

## Prepare The LiDAR Dataset
Before running the system it is necessary to record a scene with a LiDAR sensor where at least one object crosses a predefined ROI. This data must be saved in the point cloud format of XYZL, where L is the luminosity. Then, this data is used as input of the system.

## Configuration parameters
First of all, some parameters can be configured in ./python_scripts/config.txt file. Whit these parameters can be modified some functionality of the code such as selection of the projection plane (XZ or XY) in the object detection stage, capturing a new ground truth background scene used in the object detection stage, defining the location of the LiDAR data, or show some debugging information such as logs and images used during the object detection stage.

## Usage
First of all, it is necessary to compile the C++ files into the deployed machine to generate the shared object file. This file will be then called by the python script. The compilation is performed by using the makefile inside Debug directory.


```bash
cd ./Debug
make clean
make all
```

The system is executed using a python script. The object detection stage is encapsulated in a C++ function executed in the python script. To execute the system:

```bash
cd ./python_scripts
python3 python run_system.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!-- ## License
[MIT](https://choosealicense.com/licenses/mit/) -->

## Citation

If you use any part of the code from this project, please cite our paper:

@article{wisultschew20213d,
  title={3D-LIDAR based object detection and tracking on the edge of IoT for railway level crossing},
  author={Wisultschew, Cristian and Mujica, Gabriel and Lanza-Gutierrez, Jose Manuel and Portilla, Jorge},
  journal={IEEE Access},
  volume={9},
  pages={35718--35729},
  year={2021},
  publisher={IEEE}
}
