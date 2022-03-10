import ctypes
import csv
import os
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import platform
import collections
import operator

'''
source /home/cristian/virtualenvs/coral/bin/activate
python test_accuracy.py
'''
# Configuration parametres
print_results = True
load_results = False
labeled_data_dir = "/home/cristian/Desktop/LabelPointCloud/datasets/ObjDet_generated_dataset_Austria/"
total_processed_frames = 732
start_frame_ID = 539

voxels_files_dir = "../c_algorithm_outputs/detected_objects_in_voxels/"
object_detection_outputs = "../c_algorithm_outputs/object_detection_outputs.csv"
tflite_saved_file = "./tflite_model/model_edgetpu.tflite"
config_file_dir = "./config.txt"
results_file_dir = "./tools/results.txt"
voxel_size = 16
image_size = [400, 200] # W X H

def main():
    performance_metrics = load_results_file(load_results, results_file_dir)
    roi_axes_limits = read_axes_limit_from_config(config_file_dir)
    writein_object_detection_outputs(frame_ID = start_frame_ID, is_first_frame=False)
    libc = ctypes.CDLL("../Debug/libTWS_V2.1_so_Simu_ubuntu.so")

    # Load network into Coral
    init_time = time.time()
    interpreter = make_interpreter(tflite_saved_file)
    interpreter.allocate_tensors()
    loadnw_time = time.time() - init_time
    if print_results == True:
        print("Load Network time: ", 1000*loadnw_time, " ms")
    deleteVoxelFiles()

    for frame_ID in range(start_frame_ID, total_processed_frames):
        predicted_classes = []
        # Run obj detection & tracking algotithm
        out = libc.main()
        obj_det_box_Xc, obj_det_csv_vel, obj_det_csv_area = loadcsv(object_detection_outputs, roi_axes_limits)
        voxel_files_names = os.listdir(voxels_files_dir)
        if os.listdir(voxels_files_dir) != []:
            for voxels_file in voxel_files_names:
                Voxel = read_voxel_file(voxels_files_dir + voxels_file)
                predicted_classes_aux = make_inference(Voxel, interpreter)
                predicted_classes.append(predicted_classes_aux)
        else:
            if print_results == True:
                print("No objects")
        deleteVoxelFiles()
        # Calculate performance metrics to evaluate the model
        HWLXcYcZc, classes_in_scene = read_object_point_colud(frame_ID, labeled_data_dir)
        real_classes = object_classes_inside_ROI(HWLXcYcZc, classes_in_scene, roi_axes_limits)
        performance_metrics["total_real_objects"] += len(real_classes)
        performance_metrics["total_predicted_objects"] += len(predicted_classes)
        performance_metrics["total_classified_objects"] += min(len(real_classes),len(predicted_classes))
        correct_predictions = calculate_correct_predictions(HWLXcYcZc, real_classes, obj_det_box_Xc, predicted_classes)
        performance_metrics["objects_classified_correctly"] += correct_predictions
        performance_metrics = calculate_performance_metrics(real_classes, predicted_classes, correct_predictions, performance_metrics)
        if (load_results == True):
            write_results_in_file(results_file_dir, performance_metrics)
        if print_results == True:
            # print("HWLXcYcZc: ",HWLXcYcZc)
            # print("obj_det_box_Xc: ",obj_det_box_Xc)
            print("real_classes: ", real_classes)
            print("predicted_classes: ", predicted_classes)
            print("correct_predictions: ", performance_metrics["objects_classified_correctly"])
            print("performance_metrics[total_real_objects]: ", performance_metrics["total_real_objects"])
            print("performance_metrics[total_predicted_objects]: ", performance_metrics["total_predicted_objects"])
            print("performance_metrics[total_classified_objects]: ", performance_metrics["total_classified_objects"])
            print("Accuracy: ", (performance_metrics["objects_classified_correctly"]/performance_metrics["total_classified_objects"])*100, "%")
            print("performance_metrics: ", performance_metrics)


def write_results_in_file(results_file_dir, performance_metrics):
    row_cnt = 0
    object_detection_outputs_aux = "./tools/results_aux.txt"
    with open(results_file_dir, 'r') as f_in, open(object_detection_outputs_aux, 'w') as f_out:
        header = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in header:
            row_cnt += 1
            if (row_cnt == 1):
                row[0] = '// Results for calculate the accuracy'
            if (row_cnt == 2):
                row[0] = 'TP = ' + str(int(performance_metrics["TP"]))
            if (row_cnt == 3):
                row[0] = 'TN = ' + str(int(performance_metrics["TN"]))
            if (row_cnt == 4):
                row[0] = 'FP = ' + str(int(performance_metrics["FP"]))
            if (row_cnt == 5):
                row[0] = 'FN = ' + str(int(performance_metrics["FN"]))
            if (row_cnt == 6):
                row[0] = 'total_real_objects = ' + str(int(performance_metrics["total_real_objects"]))
            if (row_cnt == 7):
                row[0] = 'total_predicted_objects = ' + str(int(performance_metrics["total_predicted_objects"]))
            if (row_cnt == 8):
                row[0] = 'total_classified_objects = ' + str(int(performance_metrics["total_classified_objects"]))
            if (row_cnt == 9):
                row[0] = 'objects_classified_correctly = ' + str(int(performance_metrics["objects_classified_correctly"]))

            writer.writerow(row)
    os.system("mv "+ object_detection_outputs_aux + " " + results_file_dir)

def load_results_file(load_results, results_file_dir):
    performance_metrics = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "Sensitivity": 0,
        "Specificity": 0,
        "Precision": 0,
        "F1": 0,
        "total_real_objects": 0,
        "total_predicted_objects": 0,
        "total_classified_objects": 0, # to claculate accuracy, when we have false negative or false positive results, they dont count for the accuracy calculation
        "objects_classified_correctly": 0
    }
    if load_results == True:
        with open(results_file_dir, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if (row != []):
                    if (row[0][0:2] == "TP"):
                        performance_metrics["TP"] = float(row[0].split(" ")[2])
                    if (row[0][0:2] == "TN"):
                        performance_metrics["TN"] = float(row[0].split(" ")[2])
                    if (row[0][0:2] == "FP"):
                        performance_metrics["FP"] = float(row[0].split(" ")[2])
                    if (row[0][0:2] == "FN"):
                        performance_metrics["FN"] = float(row[0].split(" ")[2])
                    if (row[0][0:18] == "total_real_objects"):
                        performance_metrics["total_real_objects"] = float(row[0].split(" ")[2])
                    if (row[0][0:23] == "total_predicted_objects"):
                        performance_metrics["total_predicted_objects"] = float(row[0].split(" ")[2])
                    if (row[0][0:24] == "total_classified_objects"):
                        performance_metrics["total_classified_objects"] = float(row[0].split(" ")[2])
                    if (row[0][0:28] == "objects_classified_correctly"):
                        performance_metrics["objects_classified_correctly"] = float(row[0].split(" ")[2])
        csvFile.close()
    return performance_metrics


def calculate_performance_metrics(real_classes, predicted_classes, correct_predictions, performance_metrics):
    # If there are not objects in the ROI and there are no predicted objects TN is increased
    if (len(real_classes) == len(predicted_classes) and len(real_classes) == 0):
        performance_metrics["TN"] += 1
    # If there are objects in the ROI and/or there are predicted objects, TP, FN and/or FP are increased
    else:
        if (len(real_classes) == len(predicted_classes)):
            performance_metrics["TP"] += len(predicted_classes)
        if (len(real_classes) > len(predicted_classes)):
            performance_metrics["TP"] += len(predicted_classes)
            performance_metrics["FN"] += len(real_classes) - len(predicted_classes)
        if (len(real_classes) < len(predicted_classes)):
            performance_metrics["TP"] += len(real_classes)
            performance_metrics["FP"] += len(predicted_classes) - len(real_classes)

    if (performance_metrics["TP"] != 0 and performance_metrics["FN"]!= 0):
        performance_metrics["Sensitivity"] = 100*performance_metrics["TP"] / (performance_metrics["TP"]+performance_metrics["FN"])
        performance_metrics["Specificity"] = 100*performance_metrics["TN"] / (performance_metrics["TN"]+performance_metrics["FN"])
        performance_metrics["Precision"] = 100*performance_metrics["TP"] / (performance_metrics["TP"]+performance_metrics["FP"])
        performance_metrics["F1"] = 2* ((performance_metrics["Precision"]*performance_metrics["Sensitivity"]) / (performance_metrics["Precision"]+performance_metrics["Sensitivity"]))
    return performance_metrics


# This function calculate the correct answers by comparing the real lables withs
# the predocted ones. To assure that the real and predicted labels are form the
# same object, the objects with similar bounding box X coordinated between real
# and predicted aretaking as the same
def calculate_correct_predictions(HWLXcYcZc, real_classes, obj_det_box_Xc, predicted_classes):
    correct_predictions = 0
    if (len(real_classes) >= len(predicted_classes)):
        ID_dist_min = np.zeros(len(predicted_classes),dtype=np.int8)
        for ID_pred_obj in range(len(predicted_classes)):
            distance_min = 1000
            for ID_real_obj in range(len(real_classes)):
                distance_between_objects = abs(HWLXcYcZc[ID_real_obj][3] - obj_det_box_Xc[ID_pred_obj])
                if distance_between_objects < distance_min:
                    distance_min = distance_between_objects
                    ID_dist_min[ID_pred_obj] = ID_real_obj

        if (len(predicted_classes) == 1):
            if predicted_classes[0] == real_classes[ID_dist_min[0]]:
                correct_predictions += 1
        else:
            for i in range(len(predicted_classes)):
                if predicted_classes[i] == real_classes[ID_dist_min[i]]:
                    correct_predictions += 1

    if (len(predicted_classes) > len(real_classes)):
        ID_dist_min = np.zeros(len(real_classes),dtype=np.int8)
        for ID_real_obj in range(len(real_classes)):
            distance_min = 1000
            for ID_pred_obj in range(len(predicted_classes)):
                distance_between_objects = abs(HWLXcYcZc[ID_real_obj][3] - obj_det_box_Xc[ID_pred_obj])
                if distance_between_objects < distance_min:
                    distance_min = distance_between_objects
                    ID_dist_min[ID_real_obj] = ID_pred_obj
        if (len(real_classes) == 1):
            if real_classes[0] == predicted_classes[ID_dist_min[0]]:
                correct_predictions += 1
        else:
            for i in range(len(predicted_classes)):
                if (ID_dist_min!=[]):
                    if real_classes[i] == predicted_classes[ID_dist_min[i]]:
                        correct_predictions += 1

    return correct_predictions



def object_classes_inside_ROI(HWLXcYcZc, classes, roi_axes_limits):
    classes_inside_ROI = []
    bounding_box_margin = 0.2 # Bounding box margin not counted as inside the ROI in m
    for object_ID in range(len(HWLXcYcZc)):
        # If the limit of the bounding box with a margin is inside the ROI there is counted as an object
        if HWLXcYcZc[object_ID,3] >= (roi_axes_limits[0][0] - HWLXcYcZc[object_ID,0]/2 + bounding_box_margin) and HWLXcYcZc[object_ID,3] <= (roi_axes_limits[0][1] + HWLXcYcZc[object_ID,0]/2 - bounding_box_margin):
            if HWLXcYcZc[object_ID,4] >= (roi_axes_limits[1][0] - HWLXcYcZc[object_ID,1]/2 + bounding_box_margin) and HWLXcYcZc[object_ID,4] <= (roi_axes_limits[1][1] + HWLXcYcZc[object_ID,1]/2 - bounding_box_margin):
                if HWLXcYcZc[object_ID,5] >= roi_axes_limits[2][0] and HWLXcYcZc[object_ID,5] <= roi_axes_limits[2][1]:
                    classes_inside_ROI.append(classes[object_ID])

    return classes_inside_ROI


def read_object_point_colud(frame_ID, labeled_data_dir):
    csv_dir = labeled_data_dir + "labels/label_" + str(frame_ID) + ".txt"
    num_points = 0
    for row in open(csv_dir):
        num_points += 1
    HWLXcYcZc = np.zeros((num_points, 6))
    classes = []
    cnt_csv = 0
    with open(csv_dir, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=' ')
        for row in reader:
            HWLXcYcZc[cnt_csv,0] = row[8]
            HWLXcYcZc[cnt_csv,1] = row[9]
            HWLXcYcZc[cnt_csv,2] = row[10]
            HWLXcYcZc[cnt_csv,3] = row[11]
            HWLXcYcZc[cnt_csv,4] = row[12]
            HWLXcYcZc[cnt_csv,5] = row[13]
            classes.append(row[0])
            cnt_csv += 1
    csvFile.close()
    return HWLXcYcZc, classes


def read_lidar_frame_point_cloud(frame_ID, labeled_data_dir):
    csv_dir = labeled_data_dir + "frames/frame_" + str(frame_ID) + ".csv"
    num_points = 0
    for row in open(csv_dir):
        num_points += 1
    XYZL = np.zeros((num_points, 4))
    cnt_csv = 0
    with open(csv_dir, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            XYZL[cnt_csv,0] = row[0]
            XYZL[cnt_csv,1] = row[1]
            XYZL[cnt_csv,2] = row[2]
            XYZL[cnt_csv,3] = row[3]
            cnt_csv += 1
    csvFile.close()
    return XYZL

def read_axes_limit_from_config(config_file_dir):
    roi_axes_limits = [[0, 0 ],[0, 0],[0, 0]]
    with open(config_file_dir, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if (row != []):
                if (row[0][0:4] == "XMIN"):
                    roi_axes_limits[0][0] = float(row[0].split(" ")[2])
                if (row[0][0:4] == "XMAX"):
                    roi_axes_limits[0][1] = float(row[0].split(" ")[2])
                if (row[0][0:4] == "YMIN"):
                    roi_axes_limits[1][0] = float(row[0].split(" ")[2])
                if (row[0][0:4] == "YMAX"):
                    roi_axes_limits[1][1] = float(row[0].split(" ")[2])
                if (row[0][0:4] == "ZMIN"):
                    roi_axes_limits[2][0] = float(row[0].split(" ")[2])
                if (row[0][0:4] == "ZMAX"):
                    roi_axes_limits[2][1] = float(row[0].split(" ")[2])
    csvFile.close()
    return roi_axes_limits


def make_inference(Voxel, interpreter):
    top_k = 1
    threshold = 0.0

    init_time = time.time()
    set_input(interpreter, Voxel)
    interpreter.invoke()
    output = get_output(interpreter, top_k, threshold)
    inf_time = time.time() - init_time
    predicted_class = output[0][0]
    predicted_score = output[0][1]
    if print_results == True:
        print("Inference time: ", 1000*inf_time, " ms")
    return labels[predicted_class]


def deleteVoxelFiles():
    for f in os.listdir(voxels_files_dir):
        os.remove(os.path.join(voxels_files_dir, f))


def read_voxel_file(voxels_file):
    row_cnt = 1
    Voxel = []
    with open(voxels_file, 'r') as file:
        header = csv.reader(file, delimiter=',')
        for row in header:
            if( row_cnt % (voxel_size + 1) != 0) and row[0]!='EOD':
                for col in range(voxel_size):
                    Voxel.append(float(row[col]))
            row_cnt += 1
        Voxel = np.array(Voxel)
        Voxel = Voxel*255.0
        Voxel = Voxel.astype(np.uint8)
        Voxel = Voxel.reshape(16, 16, 16)
    return Voxel


def writein_object_detection_outputs(frame_ID=40, is_first_frame=False):
    row_cnt = 0
    object_detection_outputs_aux = "../c_algorithm_outputs/object_detection_outputs_aux.csv"
    with open(object_detection_outputs, 'r') as f_in, open(object_detection_outputs_aux, 'w') as f_out:
        header = csv.reader(f_in, delimiter=',')
        writer = csv.writer(f_out, delimiter=',')
        for row in header:
            row_cnt += 1
            if (row_cnt == 1 and is_first_frame == True):
                row[1] = '1'
            if (row_cnt == 4):
                row[0] = str(frame_ID)
            writer.writerow(row)
    os.system("mv "+ object_detection_outputs_aux + " " + object_detection_outputs)


def loadcsv(csvdata, roi_axes_limits):
    box_Xc = []
    vel = []
    area = []
    cnt_csv = 0
    with open(csvdata, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if cnt_csv == 0:
                num_box = int(np.shape(row)[0]) - 2
                for x in range (2,len(row) - 1):
                    box_Xc.append(float(row[x]))
            if cnt_csv == 1:
                for x in range (0, num_box - 1):
                    vel.append(float(row[x]))
            if cnt_csv == 2:
                for x in range (0, num_box - 1):
                    area.append(float(row[x]))
            cnt_csv += 1
    csvFile.close()
    # Convert XC from pixel dimensions to meters
    correctorX = (abs(roi_axes_limits[0][0]) + abs(roi_axes_limits[0][1]))/image_size[0]
    for i in range(len(box_Xc)):
        box_Xc[i] = box_Xc[i]*correctorX + roi_axes_limits[0][0]

    return box_Xc, vel, area


# Coral edgeTPU functions and constants
def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  batch, height, width, channels = interpreter.get_input_details()[0]['shape']
  return batch, width, height, channels


def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter):
  """Returns dequantized output tensor."""
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())
  scale, zero_point = output_details['quantization']
  return scale * (output_data - zero_point)


def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data


def get_output(interpreter, top_k=1, score_threshold=0.0):
  """Returns no more than top_k classes with score >= score_threshold."""
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)


labels = [ "Car", "Pedestrian", "Truck", "Cyclist"]
Class = collections.namedtuple('Class', ['id', 'score'])
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../') # Change the dir for the correct working of the C++ functions
    main()
