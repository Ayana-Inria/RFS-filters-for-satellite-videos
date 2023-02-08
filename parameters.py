import os


def get_parameters(dataset_name="WPAFB09", AOI=2):
    parameters = {"name": dataset_name}
    AOI_dir = "AOI_" + str(AOI).zfill(2)

    parameters["output_path"] = os.path.join("OUTPUTS", dataset_name, AOI_dir)

    if dataset_name == "WPAFB09":
        dataset_name = "AFRL_Dataset"
        tau = 1
        d_gt = 10
        Po = 5
        Q = 10
    elif dataset_name == "WPAFB09_Highres":
        dataset_name = "AFRL_Highres"
        tau = 1.5
        d_gt = 20
        Po = 15
        Q = 15
    else:
        raise("Dataset Not Implemented")
    # Directory Parameters
    # C:/Users/caguilar/Desktop/aguilar-camilo/Codes
    parameters["data_path"] = "dataset/" + dataset_name + "/" + AOI_dir + "/INPUT_DATA/stabilized_data"
    # parameters["data_path"] = "C:/Users/caguilar/Desktop/aguilar-camilo/Codes/FRONTIERS/dataset/AIRBUS/AOI_02/INPUT_DATA/stabilized_data"
    parameters["measurement_path"] = "dataset/" + dataset_name + "/" + AOI_dir + "/GT/labels"
    parameters["gt_csv_path"] = "dataset/" + dataset_name + "/" + AOI_dir + "/GT/transformed_object_states_moving_objects_only.csv"

    # parameters['dictionary_path'] = "../FRONTIERS/dataset/" + dataset_name + "/TRAINING_CNNS/CENTER_NET/trained_w_40_41_42_moving_objects_5fr_15_pxs/model_last.pth"
    # parameters['dictionary_path'] = "../FRONTIERS/dataset/" + dataset_name + "/TRAINING_CNNS/CENTER_NET/trained_w_40_41_42_moving_objects_5fr_15_pxs/model_last.pth"
    parameters['dictionary_path'] = "C:/Users/caguilar/Downloads/CenterTrack-master(2)/CenterTrack_og/src/trained_w_40_41_42_V2_dla34/model_last.pth"

    # Filter Parameters
    parameters['P_D'] = 0.8 #0.7  # Probability of detection
    parameters['R'] = 1     # Measurement covariance
    parameters['tau'] = tau   # Sampling period
    parameters['Q'] = Q     # Motion covariance
    parameters['Po'] = Po    # Birth covariance

    # Metrics Parameter
    parameters['c'] = d_gt
    return parameters
