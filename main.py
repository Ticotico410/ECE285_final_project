#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Chenbin Yu/ Riqian Hu

from ultralytics import YOLO
import argparse
import os
import sys
from pathlib import Path
import shutil

my_traindir = ''


def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"The file has been successfully copied from {source_path} to {destination_path}")
    except Exception as e:
        print(f"An error occurred when copying the file: {e}")


def find_latest_folder_with_keyword(directory, keyword):
    # Get the list of all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Obtain the creation time and name of each folder
    folder_creation_times = [(folder, os.path.getctime(os.path.join(directory,
                                                                    folder))) for folder in folders]

    # Sort by creation time
    folder_creation_times.sort(key=lambda x: x[1], reverse=True)

    # Traverse the sorted list of folders and find the folders containing the specified keywords
    for folder, _ in folder_creation_times:
        if keyword in folder:
            return os.path.join(directory, folder)

    # If no matching folder is found, return None
    return None

# print(metrics.box.map)  # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', type=str, default="runs/yolov8l/train/weights/yolov8l_best.pt", help='model name')
    parser.add_argument('--mode', type=str, default='test', help='"default_value"')
    args = parser.parse_args()
    last_slash_index = args.model.rfind('/')
    # print(last_slash_index)
    if args.model[-2:] == 'pt':
        model = YOLO(args.model)  # build a new model from scratch
        experiment_dir = args.model[:-3] if last_slash_index == -1 else args.model[last_slash_index + 1:-8]
    else:
        model = YOLO(args.model).load('yolov8l.pt')
        experiment_dir = args.model[:-5]

    directory_to_search = f"./runs/{experiment_dir}"
    if args.mode == 'test':

        metrics = model.val(data="myVisDrone.yaml", split='val', experiment_dir=experiment_dir,
                            save_txt=True, save_json=True)  # evaluate model performance on the validation set
        print(metrics.box.maps)
        print(metrics)
        my_testdir = find_latest_folder_with_keyword(directory_to_search, "val")

        copy_file('test.log', fr"{my_testdir}/test.log")
        print(f"The latest folder in {directory_to_search} is: {my_testdir}")
        with open(f'{my_testdir}/myresults.txt', 'w') as f:
            f.write(f"weight file：{args.model}\n ")
            f.write('total map50:' + str(metrics.box.map50) + '\n')
            f.write('total map75:' + str(metrics.box.map75) + '\n')
            f.write('total p：'+str(metrics.box.mp)+'\n')
            f.write('total r：'+str(metrics.box.mr)+'\n')
            f.write('total map50-95:' + str(metrics.box.map) + '\n')
            f.write('each map50:' + str(metrics.box.ap50) + '\n')
            f.write('each map50-95:' + str(metrics.box.ap) + '\n')
            f.write('each p:'+str(metrics.box.p)+'\n')

            f.write('speed\n')
            f.write('pre: ' + str(metrics.speed['preprocess']) + 'ms\n')
            f.write('inference: ' + str(metrics.speed['inference']) + 'ms\n')
            f.write('post: ' + str(metrics.speed['postprocess']) + 'ms\n')
            f.write('loss: ' + str(metrics.speed['loss']) + 'ms\n')
            f.write('fps: ' + str(1000/(metrics.speed['preprocess']+metrics.speed['inference']+metrics.speed['postprocess']+metrics.speed['loss'])) + '\n')
            f.write('confusion_matrix:'+str(metrics.confusion_matrix) + '\n')
            f.write(str(metrics))

    elif args.mode == 'train':
        model.train(data="myVisDrone.yaml", epochs=1, batch=4, task='detect', experiment_dir=experiment_dir)


if __name__ == '__main__':
    main()
