import src.data_utils.data_reader as data_reader
import numpy as np
import torch
import os

# TBD Filter Inports
import src.RFSs.TBD_glmb.tbd_helper_functions as smc_fns
import src.RFSs.TBD_glmb.glmb_tbd as TBD_GLMB_MAIN
import src.RFSs.RFSs_io as rfs_io

# Normal Filter Imports
import src.RFSs.GM_RFSs.gm_phd_main as GM_PHD_MAIN
import src.RFSs.GM_RFSs.glmb as GLMB_MAIN
import src.OTHERS.sort as SORT_MAIN

# CNN Imports
# from src.CNNs_new.CT_OD2 import generate_pre_heatma_from_cnn_detections
# from src.CNNs_new.CT_OD2 import create_detector
import time


from parameters import get_parameters

import cv2



def track_with_RFS_filter():
    measurement_name = "GT" # ["GT", "CNN"]
    filter_name = "GM_PHD" #"GM_PHD" # ["GLMB" , "GLMB_TBD" , "SORT"]

    # parameters
    parameters = get_parameters(dataset_name="WPAFB09", AOI=2)
    adaptive_birth = True
    cnn_assisted_birth = True
    labels_double_checking = True

    parameters['filter'] = filter_name

    # Setting Output Direcories
    output_directory = os.path.join(parameters['output_path'], filter_name, measurement_name)
    if adaptive_birth: output_directory = output_directory + "_b++"
    if cnn_assisted_birth: coutput_directory = output_directory + "_bcn++"
    print("Saving results at: \n{}".format(output_directory))

    # Filter Setup
    if filter_name == "GLMB":
        model = GLMB_MAIN.Model(parameters)
    else:
        model = smc_fns.Model(parameters)

    # Filter can be either GM-PHD (fast-good) or GLMB (slow-very good)
    rfs_filter_instance, filter_update_function = get_filter(filter_name)

    # Variables for main loop
    X = []
    X_gt = []
    max_label = 1
    k = 0
    k_final = data_reader.get_number_of_images(parameters["data_path"])
    frame_inference_dictionary = {}
    zk_prev = torch.tensor([])

    # Create Object Detector

    # We read one image to know the image size. We need multiples of 32 for the CNNs
    sample_image = data_reader.read_directory_lazy_specific_index(parameters["data_path"], 0)[:, :, 0]  # [w, h, 1]
    rows = rfs_io.highestmultiple(sample_image.shape[0], k=32)
    cols = rfs_io.highestmultiple(sample_image.shape[1], k=32)

    detection_time = 0
    if measurement_name == 'CNN':
        detector = create_detector(parameters, rows, cols)
        gt_objects = data_reader.read_csv(parameters["gt_csv_path"])
    elif measurement_name == 'GT':
        gt_objects = data_reader.read_csv(parameters["gt_csv_path"])

    # CREATE VECTOR FIELD (for adaptive birth)
    birth_field = torch.zeros((rows, cols, 3))
    frame_time_s = 1000
    Total_time = 0.01

    while k < k_final:
        print("Reading Frame {}. FPS: {}".format(k + 1, 1.0 / (float(frame_time_s) + 0.0000001)))
        image_k = data_reader.read_directory_lazy_specific_index(parameters["data_path"], k)[:rows, :cols, 0]  # [w, h, 1]

        # Force image to be 3 channels
        image_k = np.stack((image_k, image_k, image_k), axis=2).astype(np.uint8) if len(image_k.shape) == 2 else image_k
        if image_k.dtype == np.uint16:
            # change format to uint8
            image_k = image_k.astype(np.uint8)

        # Get Measurements/Likelihood
        # We are tracking by detection so we can use GT to calibrate
        if "GT" in measurement_name:
            # zk is a 4 dim vector (pi, pj, w, h).
            # Because we deal with tiny objects, w, h is hardcoded to 5
            zk, X_gt_k = data_reader.get_gt_Z_and_Xk(gt_objects[k, :], max_dims=[rows, cols])
            z_gt = zk
            zk = torch.tensor(zk.astype(np.float32), device=rfs_filter_instance.device)

            # The CNN also outputs a 6 dim vector (pi, pj, w, h, vx, vy)
            zk_6, X_gt_k = data_reader.get_gt_Z_and_Xk(gt_objects[k, :], max_dims=[rows, cols], vel=True)
            zk_6 = torch.tensor(zk_6.astype(np.float32), device=rfs_filter_instance.device)
            zk_prev = zk

            # X_gt_k is a dictionary with each key=target label value=[px, py, w, h, ...]
            X_gt_k[-999] = k + 1

            # We store each results as a list of N dims, where N is the number of frames
            X_gt.append(X_gt_k)

            pre_hm = np.zeros((1, 1, image_k.shape[0], image_k.shape[1]))

            # When training the network, I read it is good to add noise to the detections.
            if False:
                zk, zk_6 = data_reader.add_noise_to_zk(zk, zk_6=zk_6, FP_rate=0.3, FN_rate=0.2)

            frame_inference_dictionary['zk_4'] = zk
            frame_inference_dictionary['zk_prev'] = zk_prev
            frame_inference_dictionary['zk_6'] = zk_6

        elif measurement_name == 'CNN':
            z_gt, X_gt_k = data_reader.get_gt_Z_and_Xk(gt_objects[k, :], max_dims=[rows, cols])
            z_gt_m1, X_gt_k_m1 = data_reader.get_gt_Z_and_Xk(gt_objects[max(k - 1, 0), :], max_dims=[rows, cols])
            results = []
            for i in range(z_gt_m1.shape[0]):
                bb = {}
                bb['ct'] = z_gt_m1[i, np.array([1, 0])].astype(np.int32)
                bb['score'] = 1.0
                results.append(bb)

            # GAUSSIAN MIXTURE TYPE OF PRE
            pre_hm = generate_pre_heatma_from_cnn_detections(image_k.shape[0], image_k.shape[1], frame_inference_dictionary, score_thres=0.3)
            temp_time = time.time()
            frame_inference_dictionary = detector.run(image_k.astype(np.float32) / 255.0)
            detection_time = time.time() - temp_time
            frame_inference_dictionary['zk_prev'] = zk_prev
            zk_prev = frame_inference_dictionary['zk_6']
        else:
            raise ValueError('No experiment Implemented')

        # Filter Function
        debug_dictionary = {}
        debug_dictionary['image_k'] = image_k
        debug_dictionary['zs'] = frame_inference_dictionary['zk_4']
        debug_dictionary['z_gt'] = z_gt
        debug_dictionary['pre_hm'] = pre_hm

        frame_inference_dictionary['cnn_birth'] = cnn_assisted_birth
        start_time = time.time()
        rfs_filter_instance, max_label, extras_dictionary = filter_update_function(rfs_filter_instance, model,
                                                                                   frame_inference_dictionary,
                                                                                   birth_field,
                                                                                   max_label,
                                                                                   debug_dictionary)
        frame_time_s = time.time() - start_time + detection_time
        Total_time += frame_time_s
        average_FPS = float(k + 1) / float(Total_time)
        print(f"Average FPS {average_FPS:.2f}")
        # Frame Detections
        if filter_name == "GM_PHD":
            Xk = rfs_filter_instance.extract_estimates(super_labels=labels_double_checking)
        else:
            Xk = rfs_filter_instance.extract_estimates()
        Xk[-999] = k + 1
        X.append(Xk)

        # Update Vector Field
        if adaptive_birth:
            birth_field = rfs_io.update_vector_field(birth_field, X, k)

        # Plot Results
        '''
        trajectories = plot_and_save_trajectories(image_k, frame_inference_dictionary, output_directory, X, k, pre_hm)
        extras_dictionary['trajectories'] = trajectories
        extras_dictionary['debug'] = [image_k] + extras_dictionary['debug'] + [trajectories]
        plot_extras_dictionary(output_directory, extras_dictionary, k)
        '''
        # Finish loops and increase k
        k = k + 1

        if k % 100 == 0:
            rfs_io.save_object_states(X, output_directory, 0, object_detector=False)
            data_reader.calculate_metrics_and_plot(parameters, output_directory, c=parameters['c'], object_detector=False, plot=True)

            rfs_io.save_object_states(X, output_directory, 0, object_detector=True)
            data_reader.calculate_metrics_and_plot(parameters, output_directory, c=parameters['c'], object_detector=True, plot=True)

    # Save Detections
    print("Saving Detections")
    rfs_io.save_object_states(X, output_directory, 0, object_detector=True)
    print("Calculating Object Detection Metrics")
    data_reader.calculate_metrics_and_plot(parameters, output_directory, c=parameters['c'], object_detector=True, plot=True)


    # Save Tracks
    print("Saving States")
    rfs_io.save_object_states(X, output_directory, FPS=average_FPS)
    print("Calculating Metrics")
    data_reader.calculate_metrics_and_plot(parameters, output_directory, c=parameters['c'], object_detector=False, plot=True)


def plot_extras_dictionary(output_directory, extras_dictionary, k, plot_all=False):
    for title in extras_dictionary.keys():
        image_to_save = extras_dictionary[title]
        if title != 'debug':
            if plot_all:
                rfs_io.save_show_cv_image(image_to_save, output_directory=output_directory + "/extra_images", data_name=title, k=k + 1)
        else:
            debug_image = create_debug_image(image_to_save)
            rfs_io.save_show_cv_image(debug_image, output_directory=output_directory, data_name=title, k=k + 1)


def get_filter(filter_name):
    if filter_name == "GM_PHD":
        rfs_filter_instance = GM_PHD_MAIN.phd_instance()
        filter_update_function = GM_PHD_MAIN.phd_tracking_test
    elif filter_name == "GLMB":
        rfs_filter_instance = GLMB_MAIN.glmb_instance()
        rfs_filter_instance.w = torch.tensor([1])
        rfs_filter_instance.n = [0]
        rfs_filter_instance.cdn = [1]
        filter_update_function = GLMB_MAIN.jointpredictupdate_glmb
    elif filter_name == "GLMB_TBD":
        rfs_filter_instance = smc_fns.glmb_smc_instance()
        filter_update_function = TBD_GLMB_MAIN.jointpredictupdate_TBD_CNN
    elif filter_name == "SORT":
        rfs_filter_instance = SORT_MAIN.Sort()
        filter_update_function = SORT_MAIN.sort_wrapper
    return rfs_filter_instance, filter_update_function

def plot_and_save_trajectories(image_k, frame_inference_dictionary, output_directory, X, k, pre_hm):
    # Plot
    rows, cols, _ = image_k.shape
    # Plot Trajectories
    # trajectories = rfs_io.plot_trajectories(X, image_k.copy(), thickness=1, trajectory_length=10)  # ;trajectories = cv2.resize(trajectories, (cols * 2, rows * 2))
    trajectories = rfs_io.plot_trajectories_w_velocities(X, image_k.copy(), thickness=1, trajectory_length=5, add_text=True)  # ;trajectories = cv2.resize(trajectories, (cols * 2, rows * 2))
    rfs_io.save_show_cv_image(trajectories, output_directory=output_directory, data_name="trajectories", k=k+1, show_image=True)
    return trajectories

    # pre_detections = np.expand_dims(phd_gaussian_test[0, 0, ...], axis=2)

def create_debug_image(list_of_images):
    num_images = len(list_of_images)
    # I want 3 cols
    num_cols = 3
    num_rows = num_images // (num_cols + 1) + 1
    list_of_rows = []
    counter = 0
    sample_image = list_of_images[0]
    for r in range(num_rows):
        row_images = []
        for c in range(num_cols):
            if counter < num_images:
                row_images.append(list_of_images[counter])
            else:
                row_images.append(np.zeros_like(sample_image))
            counter += 1
        row_images = np.concatenate(row_images, axis=1)
        list_of_rows.append(row_images)
    list_of_rows = np.concatenate(list_of_rows, axis=0)
    return list_of_rows
