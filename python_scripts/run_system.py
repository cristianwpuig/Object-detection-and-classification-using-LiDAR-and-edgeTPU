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
python run_system.py
'''
# Configuration parametres
voxels_files_dir = "../c_algorithm_outputs/detected_objects_in_voxels/"
object_detection_outputs = "../c_algorithm_outputs/object_detection_outputs.csv"
tflite_saved_file = "./tflite_model/model_edgetpu.tflite"
voxel_size = 16
total_processed_frames = 732
start_frameID = 1

def main():
    writein_object_detection_outputs(frame_ID = start_frameID, is_first_frame=False)
    libc = ctypes.CDLL("../Debug/libTWS_V2.1_so_Simu_ubuntu.so")

    # Load network into Coral
    init_time = time.time()
    interpreter = make_interpreter(tflite_saved_file)
    interpreter.allocate_tensors()
    loadnw_time = time.time() - init_time
    print("Load Network time: ", 1000*loadnw_time, " ms")
    deleteVoxelFiles()

    for x in range(0,total_processed_frames):
        # Run obj detection & tracking algotithm
        out = libc.main()
        csv_data, csv_class, csv_vel, csv_area = loadcsv(object_detection_outputs)
        print("csv_data: ", csv_data)
        print("csv_class: ", csv_class)
        print("csv_vel: ", csv_vel)
        print("csv_area: ", csv_area)

        voxel_files_names = os.listdir(voxels_files_dir)
        print("voxel_files_names: ", voxel_files_names)
        if os.listdir(voxels_files_dir) != []:
            for voxels_file in voxel_files_names:
                Voxel = read_voxel_file(voxels_files_dir + voxels_file)
                make_inference(Voxel, interpreter)
        else:
            print("No objects")
        deleteVoxelFiles()


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
    print("Predicted class: ",labels[predicted_class])
    print("Inference time: ", 1000*inf_time, " ms")


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


def loadcsv(csvdata):
    csv_data = []
    csv_class = []
    csv_vel = []
    csv_area = []
    cnt_csv = 0
    with open(csvdata, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if cnt_csv == 0:
                num_box = int(np.shape(row)[0]) - 2
                for x in range (0,len(row) - 1):
                    csv_data.append(row[x])
            if cnt_csv == 1:
                for x in range (0, num_box - 1):
                    csv_vel.append(float(row[x]))
            if cnt_csv == 2:
                for x in range (0, num_box - 1):
                    csv_area.append(float(row[x]))
            cnt_csv += 1
    csvFile.close()
    return csv_data, csv_class, csv_vel, csv_area


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


labels = [ "car", "pedestrian", "truck/bus", "cyclist/motorbike"]
Class = collections.namedtuple('Class', ['id', 'score'])
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


if __name__ == "__main__":
    # execute only if run as a script
    main()
