from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.spatial import distance_matrix
import os
import csv
from PIL import Image
import motmetrics as mm
import torch
import cv2


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


def add_noise_to_zk(zk, zk_6=None, FP_rate=0.1, FN_rate=0):
    # zk is tensor of shape (N_meas, 4)
    if zk.shape[0] == 0:
        return zk, zk
    new_zk = []

    new_zk_6 = []

    max_rows = zk[:, 0].max().cpu().numpy()
    max_cols = zk[:, 1].max().cpu().numpy()

    indexes = torch.randperm(zk.shape[0])
    for i in range(zk.shape[0]):
        meas = zk[indexes[i], :]
        r = np.random.rand()
        if r > FN_rate:
            new_zk.append(meas)
            if zk_6 is not None:
                new_zk_6.append(zk_6[i, :])

        r2 = np.random.rand()
        if r2 < FP_rate:
            if zk_6 is not None:
                fake_z = torch.tensor(np.random.rand(6), device=zk.device, dtype=zk.dtype) * 30 + torch.tensor(np.random.rand(6) * np.array([max_rows, max_cols, 5, 5, 5, 5]), device=zk.device, dtype=zk.dtype)
                new_zk.append(fake_z[[0, 1, 4, 5]])
                new_zk_6.append(fake_z)
            else:
                fake_z = torch.tensor(np.random.rand(4) * np.array([max_rows, max_cols, 5, 5]), device=zk.device, dtype=zk.dtype)
                new_zk.append(fake_z)

    new_zk = torch.stack(new_zk, dim=0)
    if new_zk_6:
        new_zk_6 = torch.stack(new_zk_6, dim=0)
    return new_zk, new_zk_6


def get_gt_Z_and_Xk(object_array, vel=False, max_dims=False):
    '''
        Gets object row, removes empty data separates each property into a 2D tuple
        returns 3 np 2D arrays of size: [N_objects x 2] and a list of labels
    '''
    Zk = []
    Xk = {}
    for i in range(1, object_array.shape[0], 6):
        label = i // 6
        px, py = [object_array[i], object_array[i + 1]]
        vx, vy = [object_array[i + 2], object_array[i + 3]]
        ax, ay = [object_array[i + 4], object_array[i + 5]]
        if(px > 0):
            if max_dims:
                if px >= max_dims[0] or py >= max_dims[1]:
                    continue
            Xk[label] = np.array([px, py, vx, vy, 5, 5])
            if vel:
                Zk.append([px, py, vx, vy, 5, 5])
            else:
                Zk.append([px, py, 5, 5])
    Zk = np.array(Zk)
    return Zk, Xk

def get_number_of_images(directory):
    '''
        returns number_of_images in a directory
    '''
    file_names = [f for f in listdir(directory) if isfile(join(directory, f))]
    num_images = len(file_names)
    return num_images


def read_directory_lazy_specific_index(directory, index):
    list_of_files = []
    file_names = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
    name = file_names[index]
    path = directory + "/" +  name
    im = Image.open(path)
    im = np.array(im, dtype=np.uint16)
    list_of_files.append(im)
    # Sequence is of the form [t, w, h] so swap axes
    sequence = np.array(list_of_files)
    sequence = np.swapaxes(sequence, 0, 1)
    sequence = np.swapaxes(sequence, 1, 2)
    return sequence

def make_gif(frames, output_directory, file_name):
    frame_one = frames[-1]
    frame_one.save(output_directory + "/" + file_name + ".gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)


def crop_images(directory_list, output_directory, reference, ks=[1], names=None):
    # function to display the coordinates of
    # of the points clicked on the image
    x = 0
    y = 0
    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False
            # draw a rectangle around the region of interest
            cv2.rectangle(img, refPt[0], refPt[1], (0, 0, 255), 2)
            cv2.imshow("image", img)


    def draw_circle(event, x, y, flags, param):
        global refCircles
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                refCircles.append((x, y))
            except:
                refCircles = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            refCircles.append((x, y))

    # reading the image
    img0_dir = directory_list[0]
    number_images = get_number_of_images(img0_dir)
    if ks is None:
        ks = np.arange(number_images)
    img = read_directory_lazy_specific_index(img0_dir, ks[0]).astype(np.uint8)
    if(len(img.shape) > 3):
        img = img[:, :, 0, :]

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_and_crop)

    # wait for a key to be pressed to exit
    circles = False
    key = cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    if not os.path.exists(output_directory + "/" + reference):
        os.makedirs(output_directory + "/" + reference)
        os.makedirs(output_directory + "/" + reference + "/large_images")
        os.makedirs(output_directory + "/" + reference + "/marked_images")

    super_large_image = None
    large_image = None

    save_indxs = [0, len(ks) // 2, len(ks) - 1]
    # list_of_frames_for_gif = len(directory_list) * [None]
    list_of_list_of_frames = []

    dir_names_dict = {}
    directory_counter = 0
    for im_dir in directory_list:
        list_of_frames = []
        dir_names = os.path.normpath(im_dir).split(os.sep)
        if names is None:
            title = dir_names[-1]
        else:
            title = names[directory_counter]
            directory_counter += 1

        dir_name = dir_names[-3] + "_" + dir_names[-2] + "_" + dir_names[-1]
        if dir_name in dir_names_dict.keys():
            dir_names_dict[dir_name] += 1
        else:
            dir_names_dict[dir_name] = 1

        if(dir_names_dict[dir_name] > 1):
            dir_name = dir_name + str(dir_names_dict[dir_name])
        print(dir_name)
        if not os.path.exists(output_directory + "/" + reference + "/" + dir_name):
            os.makedirs(output_directory + "/" + reference + "/" + dir_name)
        number = 0
        for k in ks:
            temp_image = read_directory_lazy_specific_index(im_dir, k).astype(np.uint8)
            if len(temp_image.shape) > 3:
                temp_image = temp_image[:, :, 0, :]
            cropped_im = temp_image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0], ...]

            if(cropped_im.shape[-1] == 1):
                cropped_im = np.concatenate((cropped_im, cropped_im, cropped_im), axis=2)

            white_strip = 255 * np.ones((cropped_im.shape[0], 10, 3)).astype((np.uint8))

            textSize = cv2.getTextSize(title, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1)
            text_width, text_height = textSize[0]
            offset_text = cropped_im.shape[1] // 2 - text_width // 2 - 1
            cv2.rectangle(cropped_im, (offset_text, 10 - text_height // 2 - 3), (cropped_im.shape[1] // 2 + text_width // 2 + 1, 10 + text_height // 2 + 3), color=(0, 0, 0), thickness=-1)
            cv2.putText(cropped_im, title, org=(offset_text, 20), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(255, 255, 255))

            if(number in save_indxs):
                if(number == 0):
                    large_image = np.concatenate((cropped_im, white_strip), axis=1)
                else:
                    large_image = np.concatenate((large_image, cropped_im, white_strip), axis=1)

            PIL_image = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2RGB)
            list_of_frames.append(Image.fromarray(PIL_image))
            cv2.imwrite(output_directory + "/" + reference + "/" + dir_name + "/im_" + str(number).zfill(2) + ".png", cropped_im)
            number += 1

        # Create large image that contains all plots at 3 times
        if super_large_image is None:
            super_large_image = large_image
            white_h_strip = 255 * np.ones((15, super_large_image.shape[1], 3)).astype((np.uint8))
            super_large_image = np.concatenate((super_large_image, white_h_strip), axis=0)
        else:
            super_large_image = np.concatenate((super_large_image, large_image, white_h_strip), axis=0)

        # create large image for gif
        # if super_large_image_for is None:

        # Write large image and GIF
        cv2.imwrite(output_directory + "/" + reference + "/large_images/" + dir_name + ".png", large_image)
        make_gif(list_of_frames, output_directory + "/" + reference, dir_name)

        # Append GIF to the list of frames
        list_of_list_of_frames.append(list_of_frames)

    white_h_strip = 255 * np.ones((30, super_large_image.shape[1], 3)).astype((np.uint8))
    super_large_image = np.concatenate((white_h_strip, super_large_image), axis=0)

    offset_x = cropped_im.shape[1] // 2
    # cv2.putText(super_large_image, "t:", org=(10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 0))

    for k_idx in save_indxs:
        k = ks[k_idx]
        cv2.putText(super_large_image, "k={}".format(k), org=(offset_x, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 0))
        offset_x += cropped_im.shape[1]
    cv2.imwrite(output_directory + "/" + reference + "/SUPER_LARGE_IMAGE.png", super_large_image)

    num_frames = len(list_of_frames)
    i = 0
    print("q: left. e: right. c: draw_circles_and_save")
    while True:
        counter = 0
        for list_of_images in list_of_list_of_frames:
            cv2.imshow('image{}'.format(counter), np.array(list_of_images[i]))
            cv2.setMouseCallback('image{}'.format(counter), draw_circle)
            counter += 1

        key = cv2.waitKey(100)
        # wait for a key to be pressed to exit
        if(key&0xFF == 27):
            break
        elif(key == ord('q')):
            i = i - 1
            if(i < 0):
                i = num_frames - 1
        elif(key == ord('e')):
            i += 1
            if(i == num_frames):
                i = 0
        elif key == ord('c'):
            cv2.destroyAllWindows()
            counter = 0
            for list_of_images in list_of_list_of_frames:
                temp_image = np.array(list_of_images[i])
                for c_i in range(0, len(refCircles), 2):
                    circle_diameter = max(refCircles[c_i + 1][0] - refCircles[c_i][0], refCircles[c_i + 1][1] - refCircles[c_i][1])
                    coords = (refCircles[c_i][0] + circle_diameter // 2, refCircles[c_i][1] + circle_diameter // 2)
                    temp_image = cv2.circle(temp_image, coords, circle_diameter // 2, (0, 0, 255), 2)

                if names is None:
                    title = os.path.normpath(directory_list[counter]).split(os.sep)[-1]
                else:
                    title = names[counter]

                cv2.imwrite(output_directory + "/" + reference + "/marked_images/" + title + "_{:02}.png".format(i), temp_image)
                list_of_images[i] = Image.fromarray(temp_image)
                counter += 1
        # elif key == ord('z'):
        #    refCircles = []
    # close the window
    cv2.destroyAllWindows()

    print(f"rows: {refPt[0][1]} , {refPt[1][1]}")
    print(f"cols: {refPt[0][0]} , {refPt[1][0]}")
    file_name = os.path.join(output_directory, reference, "coords.txt")
    with open(file_name, mode='w') as f:
        f.write("coords")
        f.write("ROWS")
        f.write("x1: {}".format(refPt[0][1]))
        f.write("x2: {}".format(refPt[1][1]))
        f.write("COLS")
        f.write("y1: {}".format(refPt[0][0]))
        f.write("y2: {}".format(refPt[1][0]))



def calculate_metrics_and_plot(parameters, output_directory, c=30, object_detector=False, plot=False):
    print("Calculating metrics, |Object detector: {} | Plot: {}|".format(object_detector, plot))
    gt_csv_dir = parameters["gt_csv_path"]
    if object_detector:
        inference_csv_dir = output_directory + "/object_states_OD.csv"
    else:
        inference_csv_dir = output_directory + "/object_states.csv"

    if plot:
        if object_detector:
            image_path = output_directory + "/metrics_od"
        else:
            image_path = output_directory + "/metrics_tracking"

        if not os.path.exists(image_path):
            os.makedirs(image_path)

    gt_objects = read_csv(gt_csv_dir)
    inference_objects = read_csv(inference_csv_dir)

    gt_frames = gt_objects.shape[0]
    inf_frames = inference_objects.shape[0]

    gt_counter = 0
    inf_counter = 0

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)
    acc_counter = 0

    total_fns = 0
    total_fps = 0
    total_ids = 0
    total_gts = 0
    total_tps = 0

    f1_manual = None
    while gt_counter < gt_frames and inf_counter < inf_frames:
        inf_frame_number = inference_objects[inf_counter, 0]
        gt_frame_number = gt_objects[gt_counter, 0]

        if "WPAFB09" in parameters["name"]:
            gt_frame_number -= 99

        # Make sure both gt and inf are about the same frame
        if(inf_frame_number < gt_frame_number):
            while(inf_frame_number < gt_frame_number and inf_counter < inf_frames):
                inf_counter += 1
                inf_frame_number = inference_objects[inf_counter, 0]
        elif(gt_frame_number < inf_frame_number):
            while(gt_frame_number < inf_frame_number and gt_counter < gt_frames):
                gt_counter += 1
                gt_frame_number = gt_objects[gt_counter, 0]
                if("AFRL" in parameters["name"]):
                    gt_frame_number -= 99
        else:
            # case they are the same
            pass

        if(gt_frame_number != inf_frame_number):
            print("Error with metrics calculation. Gt and Inference data does not have overlapping frames")
            return -9999, -9999, -9999
        else:
            # print("Calculating metric for frame: {}".format(gt_frame_number))
            gt_counter += 1
            inf_counter += 1

        p_gt, v_gt, a_gt, labels_gt = get_x_clean_row(gt_objects[gt_counter - 1, :])
        p_inf, v_inf, a_inf, labels_inf = get_x_clean_row(inference_objects[inf_counter - 1, :])
        N_gt = len(p_gt)
        N_inf = len(p_inf)

        if N_gt == 0 or N_inf == 0:
            continue

        # Distance Matrix
        dist_m_p = distance_matrix(p_gt, p_inf)
        rows, cols = np.where(dist_m_p > c)
        dist_m_p[rows, cols] = np.nan
        acc.update(
        labels_gt,                     # Ground truth objects in this frame
        labels_inf,                  # Detector hypotheses in this frame
        dist_m_p,     # Distances from object 1 to hypotheses 1, 2, 3
        )

        if plot:
            # GTS: 23 | INFS: 28 | MATCHES: 18 | MOT_EVENTS: 33 = 18 (TP) + 5 (FN) + 10 (FP)
            gt_ids = [i for i in acc.mot_events['OId'][acc_counter]]
            hyp_ids = [i for i in acc.mot_events['HId'][acc_counter]]
            event_type = [i for i in acc.mot_events['Type'][acc_counter]]
            acc_counter += 1

            gt_object_dict = {}
            for idx in range(len(labels_gt)):
                l = labels_gt[idx]
                p = p_gt[idx, :]
                gt_object_dict[l] = p

            hyp_object_dict = {}
            for idx in range(len(labels_inf)):
                l = labels_inf[idx]
                p = p_inf[idx, :]
                hyp_object_dict[l] = p

            image = read_directory_lazy_specific_index(parameters['data_path'], int(gt_frame_number) - 1)

            num_dims = len(image.shape)
            if num_dims < 4:
                image = np.concatenate((image, image, image), axis=2).astype(np.uint8)
            else:
                image = image[:, :, 0, :].astype(np.uint8)

            colors = [[0, 255, 0], [255, 255, 0], [0, 0, 255], [0, 255, 255]]

            num_tps = 0
            num_fps = 0
            num_fns = 0
            num_switch = 0
            for idx in range(len(event_type)):
                ev = event_type[idx]
                if(ev == 'MATCH'):
                    l = hyp_ids[idx]
                    p = hyp_object_dict[l]
                    color_idx = 0
                    num_tps += 1
                elif(ev == 'FP'):
                    l = hyp_ids[idx]
                    p = hyp_object_dict[l]
                    color_idx = 1
                    num_fps += 1
                elif(ev == 'MISS'):
                    l = gt_ids[idx]
                    p = gt_object_dict[l]
                    color_idx = 2
                    num_fns += 1
                else:
                    # SWITCH
                    l = hyp_ids[idx]
                    p = hyp_object_dict[l]
                    if object_detector:
                        color_idx = 0
                        num_tps += 1
                    else:
                        color_idx = 3
                        num_switch += 1
                cv2.circle(image, (int(p[1]), int(p[0])), 2, colors[color_idx], 1)

            total_tps += num_tps
            total_fns += num_fns
            total_fps += num_fps
            total_ids += (num_switch / 2)
            total_gts += N_gt

            mota = 1.0 - float(total_fns + total_ids + total_fps) / float(total_gts)
            f1_manual = float(total_tps) / float(total_tps + 0.5 * float(total_fps + total_fns))

            rows, cols = image.shape[0:2]
            # Using cv2.putText() method
            cv2.putText(image, 'TP: {}({})'.format(num_tps, total_tps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, colors[0], 2, cv2.LINE_AA)
            cv2.putText(image, 'FP: {}({})'.format(num_fps, total_fps), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1, colors[1], 2, cv2.LINE_AA)
            cv2.putText(image, 'FN: {}({})'.format(num_fns, total_fns), (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                1, colors[2], 2, cv2.LINE_AA)
            if not object_detector:
                cv2.putText(image, 'SWITCH: {}({})'.format(num_switch, total_ids), (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                1, colors[3], 2, cv2.LINE_AA)

                cv2.putText(image, 'MOTA: {:.2f}'.format(mota), (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                1,(255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'F1: {:.2f}'.format(f1_manual), (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                1,(255, 255, 255), 2, cv2.LINE_AA)

            cv2.imwrite(image_path + "/metrics_" + str(int(gt_frame_number)).zfill(4) + ".png", image)

    mh = mm.metrics.create()
    summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=['num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_detections', 'num_false_positives', 'num_misses', 'num_switches', 'recall', 'precision', 'mota', 'motp'],
    names=['full', 'part'],
    generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    precision = float(int(summary["precision"]['OVERALL'] * 1000)) / 1000
    recall = float(int(summary["recall"]['OVERALL'] * 1000)) / 1000
    f1 = (2 * precision * recall ) / (precision + recall + 0.0000001)

    print(strsummary)
    # print(f1)
    f1_manual = f1_manual if f1_manual else f1
    print("f1 manual:", f1_manual)
    # file = open(output_directory + "/test.txt", "w")
    if object_detector:
        file = open(output_directory + "/metrics_f1_{:.2f}.txt".format(f1_manual), "w")
    else:
        file = open(output_directory + "/metrics_mota_{:.2f}.txt".format(summary["mota"]['OVERALL']), "w")

    if total_tps:
        strsummary += "\n\n\n"
        strsummary += "TPS: {}\n".format(total_tps)
        strsummary += "FPS: {}\n".format(total_fps)
        strsummary += "FNS: {}\n".format(total_fns)
        strsummary += "Switches: {}\n".format(total_ids)
        strsummary += "Manual F1: {}\n".format(f1_manual)

    file.writelines(strsummary)
    file.writelines("\nf1: {}".format(f1))
    file.close()

if __name__ == '__main__':
    # importing the module
    import cv2
    pass
