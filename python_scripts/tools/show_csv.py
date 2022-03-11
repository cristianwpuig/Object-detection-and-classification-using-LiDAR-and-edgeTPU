import os
import sys
import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
source /home/cristian/virtualenvs/coral/bin/activate
python show_csv.py
'''

dataset_dir = "./generated_voxels/"
voxel_size = 16
def main():
    categories = ["Ped", "Car", "Tru", "Cyc" ]
    files = os.listdir(dataset_dir)
    for file in files:
        voxel = read_voxel_file(dataset_dir + file)
        plot(voxel, file)


def read_voxel_file(voxels_file):
    row_cnt = 1
    voxel = []
    with open(voxels_file, 'r') as file:
        header = csv.reader(file, delimiter=',')
        for row in header:
            if( row_cnt % (voxel_size + 1) != 0) and row[0]!='EOD':
                for col in range(voxel_size):
                    voxel.append(float(row[col]))
            row_cnt += 1
        voxel = np.array(voxel)
        voxel = voxel*255.0
        voxel = voxel.astype(np.uint8)
        voxel = voxel.reshape(16, 16, 16)
    return voxel

def plot(data, file):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data, edgecolor="k")
    plt.title(file)
    plt.show()


if __name__ == '__main__':
    sys.exit(main() or 0)
