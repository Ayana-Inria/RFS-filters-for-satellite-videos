from src.RFSs.GM_RFSs.glmb import get_birth_gm_w_birth_field, concatenate_tracks, Track_List
import src.RFSs.RFSs_io as rfs_io
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import cv2
#from src.CNNs.CT_OD import generate_pre_heatma_from_cnn_detections
#from src.CNNs.utils.image import draw_umich_gaussian

class phd_instance:
    def __init__(self, track_list=None, device=None):

        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        # track_list
        self.tt = Track_List()                      # track table for GLMB (cell array of structs for individual tracks)

        if track_list is not None:
            self.tt = track_list

        self.X_hat_tensor = torch.empty((0, 6, 0), device=device)
        self.X_hat_dict = {}
    def prune(self, threshold=0.01):
        idxkeep = (self.tt.ws > threshold).nonzero()[:, 0]
        self.tt = self.tt.index_w_tensor(idxkeep)

    def clean_labels(self, d_threshold=50):
        label_dict = self.X_hat_dict
        for idx in range(self.tt.Num_Ts):
            label = self.tt.Ls[idx][1]
            if label in label_dict.keys():
                mu = self.tt.mus[idx, :, 0]
                mu_hat_prev = label_dict[label]
                d = np.linalg.norm(mu[0:2].cpu().numpy() - mu_hat_prev[0:2])
                if d > d_threshold:
                    self.tt.Ls[idx][1] = self.tt.Ls[idx][1] + 10001
                    print("Labeled ", self.tt.Ls[idx][1], "has been cleaned")
        pass

    def inference(self, all=False):
        if all:
            n_hat = self.tt.Num_Ts
        else:
            n_hat = torch.round(self.tt.ws.sum()).cpu().int().item()

        if(n_hat == 0):
            return {}
        indexes = torch.argsort(self.tt.ws[:, 0], descending=True)
        indexes = indexes[torch.arange(n_hat).long()]
        mus = self.tt.mus[indexes][:, :, 0].cpu().numpy()
        Ps = [self.tt.Ps[i] for i in indexes]
        ws = [self.tt.ws[i].item() for i in indexes]
        Ls = [self.tt.Ls[i][1] for i in indexes]

        D_updated = {}
        D_updated['Hs'] = n_hat
        D_updated['mus'] = mus
        D_updated['Ps'] = Ps
        D_updated['Ls'] = Ls
        D_updated['ws'] = ws
        return D_updated

    def extract_estimates(self, fiter_static_objects=False, super_labels=False):
        Xk = {}
        n_hat = torch.round(self.tt.ws.sum()).cpu().int().item()

        if(n_hat == 0):
            return Xk
        indexes = torch.argsort(self.tt.ws[:, 0], descending=True)
        indexes = indexes[torch.arange(n_hat).long()]
        mus = self.tt.mus[indexes][:, :, 0].cpu().numpy()
        Ls = [self.tt.Ls[i][1] for i in indexes]

        for i in range(len(Ls)):
            if fiter_static_objects:
                v_norm = np.linalg.norm(mus[i, 2:4])
                if v_norm < 3:
                    continue

            if Ls[i] not in Xk.keys():
                Xk[Ls[i]] = mus[i, :]
            else:
                if super_labels:
                    Xk[Ls[i] + 10001] = mus[i, :]
                    self.tt.Ls[indexes[i]] = [0, Ls[i] + 10001]
        self.X_hat_tensor = self.tt.mus[indexes]
        self.X_hat_dict = Xk
        return Xk


    def extract_estimates_ll(self, fiter_static_objects=False, super_labels=True):
        Xk = {}
        n_hat = torch.round(self.tt.ws.sum()).cpu().int().item()

        if(n_hat == 0):
            return Xk
        # indexes = torch.argsort(self.tt.ws[:, 0], descending=True)
        indexes = torch.arange(self.tt.Num_Ts)
        # indexes = indexes[torch.arange(n_hat).long()]
        mus = self.tt.mus[indexes][:, :, 0].cpu().numpy()
        Ls = [self.tt.Ls[i][1] for i in indexes]
        ws = self.tt.ws[indexes]

        for i in range(len(Ls)):
            if ws[i] < 0.1:
                continue
            if fiter_static_objects:
                v_norm = np.linalg.norm(mus[i, 2:4])
                if v_norm < 3:
                    continue

            if Ls[i] not in Xk.keys():
                Xk[Ls[i]] = mus[i, :]
            else:
                if super_labels:
                    Xk[Ls[i] + 1000] = mus[i, :]
                    self.tt.Ls[indexes[i]][1] = Ls[i] + 10000

        self.X_hat_tensor = self.tt.mus[indexes]

        return Xk

    def get_cnn_like_representation(self, w_thres=0.5, velocity_thres=0.5):
        results = []
        # X_hat_tensor | [:, :, 0]
        for i in range(self.X_hat_tensor.shape[0]):
            bb = {}
            bb['ct'] = self.X_hat_tensor[i, torch.tensor([1, 0]), 0].cpu().numpy().astype(np.int32)
            bb['score'] = self.tt.ws[i]
            bb['label'] = self.tt.Ls[i][1]
            if torch.norm(self.X_hat_tensor[i, torch.tensor([2, 3]), 0], 1) > velocity_thres and self.tt.ws[i] > w_thres:
                results.append(bb)
        return {'results': results}

    def get_colored_representation(self, image, labels_to_plot):
        np.random.seed(0)
        colors = (255 * np.random.rand(200, 3)).astype(np.uint8)
        for i in range(self.tt.mus.shape[0]):
            temp = np.zeros((image.shape[0], image.shape[1]))
            label = self.tt.Ls[i][1]
            l_idx = (label - 1) % 200
            color = colors[l_idx, :]
            if label in labels_to_plot:
                z = self.tt.mus[i, torch.tensor([1, 0]), 0].cpu().numpy().astype(np.int32)
                w = self.tt.ws[i]
                draw_umich_gaussian(temp, z.astype(np.uint16), radius=3, k=1)
                for i in range(3):
                    image[:, :, i] += temp * color[i]
        return image


def phd_tracking_test(phd_update, model, frame_detections, birth_field, max_label, debug_dictionary):
    zk = frame_detections['zk_4']
    zk6 = frame_detections['zk_6']
    zk_prev = frame_detections['zk_prev']
    # SURVIVING TRACKS: ADVANCE STATE
    tt_survive = phd_update.tt
    tt_survive.kalman_prediction(model)
    zk_6 = frame_detections['cnn_birth']

    mus_pred_birth = torch.matmul(model.F, phd_update.X_hat_tensor)
    # mus_pred_birth = tt_survive.mus
    tt_birth, Z_surviving = get_birth_gm_w_birth_field(zk6, birth_field, mus_pred_birth,
                                                       current_label=max_label + 1, k=0,
                                                       initial_P=30,
                                                       distance_thres=10,
                                                       ZK6=zk_6)

    # tt_birth.kalman_prediction(model) # IF USING Zkm1!!!!!!!!!!!!!!!!!!!!!!!
    # Add the labels for each new component
    max_label += tt_birth.Num_Ts

    # PREDICTION TRACKS: CONCATENATION BIRTH + SURVIVAL
    tt_pred = concatenate_tracks(tt_birth, tt_survive)

    # GATE MEASUREMENT
    tt_pred.get_gated_measurements(zk, distance_thres=30, perform_gating=True)

    # TRACK UPDATE STEP
    m = zk.shape[0]
    tt_update = Track_List(model=model, N_init=((1 + m) * tt_pred.Num_Ts))  # [(1 + m) * Num_Ts_predict]

    # ADD MEASUREMENT UPDATED TRACKS
    tt_update.kalman_update_2(tt_pred, model, zk, PHD_FILTER=True)

    phd_update.tt = tt_update
    phd_update.prune(0.1)

    # phd_update.clean_labels()

    extras_dictionary = get_extras_dictionary_to_plot(debug_dictionary, frame_detections)
    return phd_update, max_label, extras_dictionary



def get_extras_dictionary_to_plot(debug_dictionary, frame_detections):
    extras_dictionary = {}

    image_k = debug_dictionary['image_k']
    rows, cols, _ = image_k.shape

    if 'output' in frame_detections.keys():
        hm = frame_detections['output']['hm']
    else:
        hm = torch.zeros((1, 1, rows, cols))
    hm_save = cv2.resize((hm[0, 0, ...].cpu().numpy() * 255).astype(np.uint8), (cols, rows))
    hm_save = np.stack((hm_save, hm_save, hm_save), axis=2)
    image_k_likelihood = cv2.addWeighted(hm_save, 0.5, image_k, 0.7, 0)
    measurements_gt = rfs_io.plot_measurements(np.transpose(debug_dictionary['z_gt']), image_k_likelihood, L=3, color=(0, 255, 0))
    measurements = rfs_io.plot_measurements(np.transpose(frame_detections['zk_4'].cpu().numpy()), image_k_likelihood, L=3, color=(0, 0, 255))
    try:
        measurements_w_velocities = rfs_io.plot_measurements_w_velocity(np.transpose(frame_detections['zk_6'].cpu().numpy()), image_k_likelihood, L=2, color=(0, 0, 255))
    except AttributeError:
        measurements_w_velocities = measurements

    measurements_overlapped = rfs_io.plot_measurements(np.transpose(frame_detections['zk_4'].cpu().numpy()), measurements_gt, L=2, color=(0, 0, 255))

    '''
    # Generate Previous 
    phd_gaussian_test = generate_pre_heatma_from_cnn_detections(image_k.shape[0], image_k.shape[1], phd_detections, score_thres=0.1)
    phd_gaussian = np.expand_dims(phd_gaussian_test[0, 0, ...], axis=2)
    phd_gaussian = (np.concatenate((phd_gaussian, phd_gaussian, phd_gaussian), axis=2) * 255).astype(np.uint8)
    phd_gaussian = cv2.addWeighted(phd_gaussian, 0.5, image_k, 0.7, 0)
    rfs_io.save_show_cv_image(phd_gaussian, output_directory=output_directory, data_name="phd_gaussian", k=k+1, show_image=False)
    '''
    if 'pre_hm' in debug_dictionary.keys():
        pre_hm = debug_dictionary['pre_hm']
    else:
        pre_hm = np.zeros((1, 1, rows, cols))
    pre_hm = np.expand_dims(pre_hm[0, 0, ...], axis=2)
    pre_hm = (np.concatenate((pre_hm, pre_hm, pre_hm), axis=2) * 255).astype(np.uint8)
    pre_hm = cv2.addWeighted(pre_hm, 0.5, image_k, 0.7, 0)

    extras_dictionary['likelihood_only'] = hm_save
    extras_dictionary['likelihood'] = image_k_likelihood
    extras_dictionary['likelihood_zk'] = measurements
    extras_dictionary['likelihood_z_GT'] = measurements_gt
    extras_dictionary['measurements_overlapped'] = measurements_overlapped
    extras_dictionary['measurements_w_velocities'] = measurements_w_velocities
    extras_dictionary['pre_hm'] = pre_hm

    hm_meas = rfs_io.plot_measurements(np.transpose(frame_detections['zk_4'].cpu().numpy()), hm_save * 5, L=2, color=(0, 0, 255))
    hm_meas = rfs_io.plot_measurements(np.transpose(debug_dictionary['z_gt']), hm_meas, L=1, color=(0, 255, 0))

    extras_dictionary['debug'] = [measurements_w_velocities,
                                  measurements_overlapped,
                                  pre_hm,
                                  hm_meas]

    return extras_dictionary

