import json
import csv
import numpy as np
from os import listdir
from os.path import isfile, join

import cv2

# TEST: AOIs 02, 03, 34   TRAIN: 40, 41,42; 
# TEST: AOIs 01, 40,      TRAIN: 34, 41,42;
# TEST: AOIs 04, 41,      TRAIN: 42, 34, 40

def main():

    AOI = 41
    PLOT_IMAGE_AND_LABELS = False
    AOI = str(AOI).zfill(2)
    csv_directories = ['C:/Users/caguilar/Desktop/aguilar-camilo/Codes/FRONTIERS/dataset/AFRL_Highres/AOI_' + AOI + '/GT/transformed_object_states_moving_objects_only.csv']
    img_direcotories = ['C:/Users/caguilar/Desktop/aguilar-camilo/Codes/FRONTIERS/dataset/AFRL_Highres/AOI_' + AOI + '/INPUT_DATA/stabilized_data']
    output_directory = csv_directories[0][0:-4] + ".json"
    
    # create return dictionary
    mot_format_dictionary = {}
    mot_format_dictionary['categories'] = []
    mot_format_dictionary['videos'] = []
    mot_format_dictionary['images'] = []
    mot_format_dictionary['annotations'] = []

    # CREATE CATEGORIES
    mot_format_dictionary['categories'].append({'id': 1, 'name': 'vehicle'})

    # CREATE VIDEOS AND IMAGES ENTRIES
    frame_id = 1
    label_id_counter = 1
    object_id = 1
    label_id_dictionaries = {}
    for video_idx in range(len(img_direcotories)):
        # create videos entry
        directory = img_direcotories[video_idx]
        video_string = csv_directories[video_idx]
        file_name = video_string.split('/')[-3]
        mot_format_dictionary['videos'].append({'id': video_idx + 1, 'file_name': file_name})

        # create images entry
        file_names = [f for f in listdir(directory) if isfile(join(directory, f))]
        num_files = len(file_names)
        frame_current_idx = 1

        # read tracks
        gt_objects = read_csv(csv_directories[video_idx])

        # read files
        for file_n in file_names:
            if frame_current_idx == 1:
                prev_image_id = -1
            else:
                prev_image_id = frame_id - 1

            if frame_current_idx == num_files:
                next_frame_id = -1
            else:
                next_frame_id = frame_id + 1

            mot_format_dictionary['images'].append({
                'file_name': file_n,
                'id': frame_id,
                'frame_id': frame_current_idx,
                'prev_image_id': prev_image_id,
                'next_image_id': next_frame_id,
                'video_id': video_idx + 1
                })

            if PLOT_IMAGE_AND_LABELS:
                img = cv2.imread(img_direcotories[0] + '/' + file_n)
            # CREATE ANNOTATIONS
            p_gt, v_gt, a_gt, labels_gt = get_x_clean_row(gt_objects[frame_current_idx - 1, :])

            for obj_idx in range(len(labels_gt)):
                label = labels_gt[obj_idx]

                # Find holistic label
                if label in label_id_dictionaries.keys():
                    track_id = label_id_dictionaries[label]
                else:
                    track_id = label_id_counter
                    label_id_dictionaries[label] = label_id_counter
                    label_id_counter += 1

                py, px = p_gt[obj_idx, :]
                w, h = a_gt[obj_idx, :]

                if PLOT_IMAGE_AND_LABELS:
                    img = cv2.circle(img, (px.astype(np.int16), py.astype(np.int16)), 4, (255, 255, 255), 1)

                if w == 0:
                    w = 2
                    h = 2

                mot_format_dictionary['annotations'].append({
                'id': object_id,
                'category_id': 1,
                'image_id': frame_id,
                'track_id': label,
                'bbox': [px - w // 2, py - h // 2, w, h], # px + w // 2, py + h // 2],
                'conf': 1.0
                })

                object_id += 1
            frame_current_idx += 1
            frame_id += 1

            if PLOT_IMAGE_AND_LABELS:
                cv2.imshow("window", img)
                # print(mot_format_dictionary['images'][-1])
                # print(mot_format_dictionary['annotations'][-1])
                cv2.waitKey(0)


    print("WRITTING JSON AOI {}".format(AOI))
    print("Writting at: " + output_directory)
    write_json(mot_format_dictionary, output_directory)
    print("DONE")


def read_csv(directory):
    '''
        Returns a np array of size [N_frames x N_objects * 6], each object has (px, py, vx, vy, ax, ay) components
    '''

    objects_list = []
    with open(directory) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        frame_counter = 0
        for row in spamreader:
            frame_counter += 1
            if(frame_counter == 1):
                continue
            objects_list.append(row)
    objects_array = np.array(objects_list)
    return objects_array.astype(np.double)




def get_x_clean_row(object_array):
    '''
        Gets object row, removes empty data separates each property into a 2D tuple
        returns 3 np 2D arrays of size: [N_objects x 2] and a list of labels
    '''
    list_of_ps = []
    list_of_vs = []
    list_of_as = []
    list_of_labels = []
    for i in range(1, object_array.shape[0], 6):
        label = i // 6
        px, py = [object_array[i], object_array[i + 1]]
        vx, vy = [object_array[i + 2], object_array[i + 3]]
        ax, ay = [object_array[i + 4], object_array[i + 5]]
        if(px > 0):
            list_of_ps.append([px, py])
            list_of_vs.append([vx, vy])
            list_of_as.append([ax, ay])
            list_of_labels.append(label)
    ps = np.array(list_of_ps)
    vs = np.array(list_of_vs)
    acs = np.array(list_of_as)
    return ps, vs, acs, list_of_labels


def write_json(dictionary, file_name):
    # Serializing json 
    json_object = json.dumps(dictionary, indent = 4)
      
    # Writing to sample.json
    with open(file_name, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()