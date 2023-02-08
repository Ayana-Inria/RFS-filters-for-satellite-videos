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

# CNN Imports
from src.CNNs.CT_OD import generate_pre_heatma_from_cnn_detections
from src.CNNs_new.CT_OD2 import create_detector
from src.CNNs.dataset.custom_dataset import CustomDataset
from src.CNNs.opts import opts
import time


from parameters import get_parameters

import cv2


def track_with_CNN_filter():
    experiment = "FINAL"
    parameters = get_parameters(dataset_name="WPAFB09", AOI=2)
    # parameters = get_parameters(dataset_name="Lyon", AOI=1)
    # Directory Sorting
    measurement_name = "CNN"
    filter_name = "CenterTrack"

    parameters['filter'] = filter_name
    output_directory = os.path.join(parameters['output_path'], filter_name, measurement_name)
    output_directory += experiment
    print("Results saved at: \n{}".format(output_directory))

    # Filter Setup
    X = []
    X_gt = []
    Zs = []

    max_label = 1
    k = 0
    k_final = data_reader.get_number_of_images(parameters["data_path"])
    frame_inference_dictionary = None

    gt_objects = data_reader.read_csv(parameters["gt_csv_path"])

    # Create Object Detector
    sample_image = data_reader.read_directory_lazy_specific_index(parameters["data_path"], 0)[:, :, 0]  # [w, h, 1]
    rows = rfs_io.highestmultiple(sample_image.shape[0], k=32)
    cols = rfs_io.highestmultiple(sample_image.shape[1], k=32)

    if measurement_name == 'CNN':
        detector = create_detector(parameters, rows, cols)

    # CREATE VECTOR FIELD
    birth_field = torch.zeros((rows, cols, 3))
    Total_time = 0.0001
    while k < k_final:
        print("Reading Frame {}".format(k + 1))
        # Read Image
        image_k = data_reader.read_directory_lazy_specific_index(parameters["data_path"], k)[:rows, :cols, 0]  # [w, h, 1]
        image_k = np.stack((image_k, image_k, image_k), axis=2).astype(np.uint8) if len(image_k.shape) == 2 else image_k

        # Get Measurements/Likelihood
        if measurement_name == "GT":
            frame_inference_dictionary = {}
            meas_curr = data_reader.read_directory_lazy_specific_index(parameters["measurement_path"], k)[:, :, 0]  # [w, h, 1]
            meas_curr = meas_curr[:rows, :cols]

            zk, X_gt_k = rfs_io.get_Zk_and_Gt(meas_curr)  # z is (N_meas, 4)
            zk = torch.tensor(zk.astype(np.float32), device=detector.opt.device)
            frame_inference_dictionary['zk_4'] = zk
            # Get GT
            X_gt_k[-999] = k + 1
            X_gt.append(X_gt_k)

            Zs.append(X_gt_k)
            start_time = 0
        elif measurement_name == 'CNN':
            zk, X_gt_k = data_reader.get_gt_Z_and_Xk(gt_objects[max(k -1, 0), :], max_dims=[rows, cols])
            results = []
            for i in range(zk.shape[0]):
                bb = {}
                bb['ct'] = zk[i, np.array([1, 0])].astype(np.int32)
                bb['score'] = 1.0
                results.append(bb)
            frame_inference_dictionary0 = {'results': results}

            pre_hm = generate_pre_heatma_from_cnn_detections(image_k.shape[0], image_k.shape[1], frame_inference_dictionary0, score_thres=0.3)
            # pre_hm = generate_pre_heatma_from_cnn_detections(image_k.shape[0], image_k.shape[1], frame_inference_dictionary, score_thres=0.3)
            start_time = time.time()
            frame_inference_dictionary = detector.run(image_k.astype(np.float32) / 255.0)
            frame_time_s = time.time() - start_time
            zk = frame_inference_dictionary['Xk']
            zk[-999] = k + 1
            Zs.append(zk)
        else:
            raise ValueError('No experiment Implemented')

        # Frame Detections
        Xk = frame_inference_dictionary['Xk']
        Total_time += frame_time_s
        average_FPS = float(k + 1) / float(Total_time)
        print("Average FPS", average_FPS)

        Xk[-999] = k + 1
        X.append(Xk)

        # Plot Results
        # plot_stuff(image_k, birth_field, frame_inference_dictionary, output_directory, X, k, pre_hm)

        # Finish loops and increase k
        k = k + 1

        if k > 1000:
            print("Saving Detections")
            rfs_io.save_object_states(X, output_directory, 0, object_detector=False)
            print("Calculating Object Detection Metrics")
            data_reader.calculate_metrics_and_plot(parameters, output_directory, c=5, object_detector=False, plot=True)

            print("Saving Detections")
            rfs_io.save_object_states(Zs, output_directory, 0, object_detector=True)
            print("Calculating Object Detection Metrics")
            data_reader.calculate_metrics_and_plot(parameters, output_directory, c=5, object_detector=True, plot=True)
            exit()


    # Finish
    print("Saving States")
    rfs_io.save_object_states(X, output_directory, 0)
    print("Calculating Metrics")
    data_reader.calculate_metrics_and_plot(parameters, output_directory, c=10, object_detector=False)


def plot_stuff(image_k, birth_field, frame_inference_dictionary, output_directory, X, k, pre_hm):
    # Plot
    rows, cols, _ = image_k.shape

    birth_field_to_plot = rfs_io.birth_field_plotter(birth_field, image_k)
    rfs_io.save_show_cv_image(birth_field_to_plot, output_directory=output_directory, data_name="birth_field", k=k+1, show_image=False)

    # Plot Trajectories
    trajectories = rfs_io.plot_trajectories(X, image_k.copy(), trajectory_length=20)  # ;trajectories = cv2.resize(trajectories, (cols * 2, rows * 2))
    rfs_io.save_show_cv_image(trajectories, output_directory=output_directory, data_name="trajectories", k=k+1, show_image=False)

    '''
    # Generate Previous Gaussian
    phd_gaussian_test = generate_pre_heatma_from_cnn_detections(image_k.shape[0], image_k.shape[1], phd_detections, score_thres=0.1)
    phd_gaussian = np.expand_dims(phd_gaussian_test[0, 0, ...], axis=2)
    phd_gaussian = (np.concatenate((phd_gaussian, phd_gaussian, phd_gaussian), axis=2) * 255).astype(np.uint8)
    phd_gaussian = cv2.addWeighted(phd_gaussian, 0.5, image_k, 0.7, 0)
    rfs_io.save_show_cv_image(phd_gaussian, output_directory=output_directory, data_name="phd_gaussian", k=k+1, show_image=False)
    '''
    pre_hm = np.expand_dims(pre_hm[0, 0, ...], axis=2)
    pre_hm = (np.concatenate((pre_hm, pre_hm, pre_hm), axis=2) * 255).astype(np.uint8)
    pre_hm = cv2.addWeighted(pre_hm, 0.5, image_k, 0.7, 0)
    rfs_io.save_show_cv_image(pre_hm, output_directory=output_directory, data_name="pre_hm", k=k+1, show_image=False)

    # pre_detections = np.expand_dims(phd_gaussian_test[0, 0, ...], axis=2)

    if 'output' in frame_inference_dictionary.keys():
        # Get Measurements
        measurements = rfs_io.plot_measurements_w_velocity(np.transpose(frame_inference_dictionary['zk_6'].cpu().numpy()), image_k, L=3)
        rfs_io.save_show_cv_image(measurements, output_directory=output_directory, data_name="measurements", k=k+1, show_image=False)

        hm = frame_inference_dictionary['output']['hm']
        hm_save = cv2.resize((hm[0, 0, ...].cpu().numpy() * 255).astype(np.uint8), (cols, rows))
        hm_save = cv2.addWeighted(hm_save, 0.5, image_k[:, :, 0], 0.7, 0)
        hm_save = np.expand_dims(hm_save, axis=2)
        hm_save = np.concatenate((hm_save, hm_save, hm_save), axis=2)
        rfs_io.save_show_cv_image(hm_save, output_directory=output_directory, data_name="likelihood", k=k+1, show_image=False)

        debug_image = np.concatenate((hm_save, pre_hm, measurements), axis=1)
        rfs_io.save_show_cv_image(debug_image, output_directory=output_directory, data_name="debug", k=k+1, show_image=False)

    else:
        measurements = rfs_io.plot_measurements_w_velocity(np.transpose(frame_inference_dictionary['zk_4'].cpu().numpy()), image_k, L=3)
        rfs_io.save_show_cv_image(measurements, output_directory=output_directory, data_name="measurements", k=k+1, show_image=False)
