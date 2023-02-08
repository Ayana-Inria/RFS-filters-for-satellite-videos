import src.RFSs.TBD_glmb.tbd_helper_functions as tbd_fns
import src.RFSs.RFSs_io as rfs_io
import torch.nn.functional as F
import numpy as np
import torch
import math

import cv2
import os


# rfs_filter_instance, model, frame_inference_dictionary, birth_field, max_label)
# rfs_filter_instance, model, frame_inference_dictionary, birth_field, max_label
def jointpredictupdate_TBD_CNN(glmb_smc_update, model, frame_inference_dictionary, birth_field, max_label):
    '''
        Generate next smc glmb state
    '''
    N_ptcls = 200
    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

    # Get likelihood
    likelihood = frame_inference_dictionary['output']['hm']
    likelihood = F.interpolate(likelihood, size=(birth_field.shape[0], birth_field.shape[1]), mode='bilinear')[0, 0, ...]
    # Get measurements
    zk_cnn = frame_inference_dictionary['zk_6']

    # SURVIVING TRACKS: ADVANCE STATE
    tt_survive = glmb_smc_update.tt
    tt_survive.kalman_prediction(model.F)

    # BIRTH TRACKS: MEASUREMENT DRIVEN PROPOSAL
    tt_birth, Z_surviving, max_label = glmb_smc_update.tt.get_birth_particles_smc(zk_cnn, current_label=max_label, N_particles_per_birth_object=N_ptcls, distance_thres=20)

    '''
    tt_birth, Z_surviving, max_label = RFSs_io.get_birth_particles_smc_w_velocities(Zk, glmb_smc_update.tt, current_label=max_label, N_particles_per_birth_object=N_ptcls, k=k, distance_thres=20)
    '''

    # PREDICTION TRACKS: CONCATENATION BIRTH + SURVIVAL
    tt_pred_update = tbd_fns.concatenate_tracks_smc(tt_birth, tt_survive)

    # ADD MEASUREMENT UPDATED TRACKS
    tt_pred_update.kalman_update_3(likelihood, regular_image=birth_field)
    tbd_fns.lmb2glmb(tt_pred_update, H_req=100)
    # GLMB COMPONENTS UPDATE
    glmb_smc_posterior = tbd_fns.glmb_smc_instance(track_list=tt_pred_update)
    if glmb_smc_posterior.tt.Num_Ts > 0:
        glmb_smc_posterior = tbd_fns.clean_update(glmb_smc_posterior)

    extras_dictionary = generate_plot_images(likelihood, tt_birth, tt_survive, tt_pred_update, glmb_smc_posterior, max_label)
    # extras_dictionary = {}
    return glmb_smc_posterior, max_label, extras_dictionary

def generate_plot_images(likelihood, tt_birth, tt_survive, tt_pred_update, glmb_smc_posterior, max_label):
    # plotting
    # likelihood_image = likelihood * 255
    # likelihood_image = cv2.applyColorMap(likelihood.cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

    likelihood = likelihood.cpu().numpy() * 255
    likelihood = np.expand_dims(likelihood, axis=2)
    likelihood = np.concatenate((likelihood, likelihood, likelihood), axis=2)

    birth_particle_image, birth_likelihood_image = render_glmb_tracks_particles(tt_birth, likelihood, max_label, R=1, L1=1, L2=3)
    surviving_particle_image1, surviving_likelihood_image = render_glmb_tracks_particles(tt_survive, likelihood, max_label, R=3, L1=3, L2=8, labels_to_plot=np.arange(0, 300, 3))
    surviving_particle_image2, surviving_likelihood_image = render_glmb_tracks_particles(tt_survive, likelihood, max_label, R=3, L1=3, L2=8, labels_to_plot=np.arange(0, 300, 3) + 1)
    surviving_particle_image3, surviving_likelihood_image = render_glmb_tracks_particles(tt_survive, likelihood, max_label, R=3, L1=3, L2=8, labels_to_plot=np.arange(0, 300, 3) + 2)
    surviving_particle_image4, surviving_likelihood_image = render_glmb_tracks_particles(tt_survive, likelihood, max_label, R=3, L1=3, L2=8, labels_to_plot=np.arange(0, 300, 3) + 3)
    surviving_particle_image, surviving_likelihood_image = render_glmb_tracks_particles(tt_survive, likelihood, max_label, R=3, L1=3, L2=8)
    updated_particle_image, updated_likelihood_image = render_glmb_tracks_particles(tt_pred_update, likelihood, max_label, R=3, L1=3, L2=8)
    resampled_particle_image, resampled_likelihood_image = render_glmb_tracks_particles(glmb_smc_posterior.tt, likelihood, max_label, R=1, L1=1, L2=3)

    # extras_dictionary['zk'] = zk_cnn
    extras_dictionary = {}
    extras_dictionary['updated_likelihood_image'] = likelihood
    extras_dictionary['resampled_image'] = resampled_particle_image
    extras_dictionary['surviving_image1'] = surviving_particle_image1
    extras_dictionary['surviving_image2'] = surviving_particle_image2
    extras_dictionary['surviving_image3'] = surviving_particle_image3
    extras_dictionary['surviving_image4'] = surviving_particle_image4
    extras_dictionary['birth_image'] = birth_particle_image

    # saving graphs
    '''
    rfs_io.save_show_cv_image(birth_particle_image, output_directory=output_directory + "/PARTICLES", data_name="birth_particles", k=k + 1)
    rfs_io.save_show_cv_image(surviving_particle_image, output_directory=output_directory + "/PARTICLES", data_name="surviving_particles", k=k + 1)
    rfs_io.save_show_cv_image(updated_particle_image, output_directory=output_directory + "/PARTICLES", data_name="updated_particles", k=k + 1)
    rfs_io.save_show_cv_image(resampled_particle_image, output_directory=output_directory + "/PARTICLES", data_name="resampled_particles", k=k + 1)
    '''
    return extras_dictionary

def render_glmb_tracks_particles(track_list, og_image, max_label=1, R=3, L1=5, L2=15, particles_to_plot=100, labels_to_plot=None):
    np.random.seed(0)

    # Xs is of shape [Num Components, N particles, 6]
    Xs = track_list.Xs
    if len(Xs.size()) == 0:
        return og_image, og_image
    # Ws is of shape [Num Components, N particles, 1]
    Ws = track_list.ws
    # index_w_tensor
    colors = (255 * np.random.rand(max_label, 3)).astype(np.uint8)

    image = np.copy(og_image)
    image = image.astype(np.uint8)

    likelihood_image = np.zeros((image.shape[0], image.shape[1]))
    for cmp_idx in range(Xs.shape[0]):

        indexes = torch.argsort(-Ws[cmp_idx, :, 0])[0:particles_to_plot]
        weight_norm = Ws[cmp_idx, indexes, 0].sum() + 0.0001
        label = track_list.Ls[cmp_idx][1]
        # for debugging purposes (plot only one label)
        if labels_to_plot is not None and label not in labels_to_plot:
            continue

        for particle_idx in indexes:
            pix = Xs[cmp_idx, particle_idx, 0].round().int().cpu().numpy().astype(np.int32)
            piy = Xs[cmp_idx, particle_idx, 1].round().int().cpu().numpy().astype(np.int32)

            v = np.array([Xs[cmp_idx, particle_idx, 2].cpu(), Xs[cmp_idx, particle_idx, 3].cpu()])
            phi = np.degrees(np.arctan2(v[1], v[0]) + np.pi / 2.0)
            if phi < 0:
                phi = phi + 180
            vel_dir1 = v / np.linalg.norm(v) * L1
            vel_dir2 = v / np.linalg.norm(v) * L2

            color = tuple(colors[label - 1, j].item() for j in range(3))

            pxo = (Xs[cmp_idx, particle_idx, 0] + vel_dir1[0]).round().cpu().numpy().astype(np.int32)
            pyo = (Xs[cmp_idx, particle_idx, 1] + vel_dir1[1]).round().cpu().numpy().astype(np.int32)

            pxf = (Xs[cmp_idx, particle_idx, 0] + vel_dir2[0]).round().cpu().numpy().astype(np.int32)
            pyf = (Xs[cmp_idx, particle_idx, 1] + vel_dir2[1]).round().cpu().numpy().astype(np.int32)

            likelihood = Ws[cmp_idx, particle_idx, 0] / weight_norm

            image = cv2.circle(image, (piy.item(), pix.item()), radius=R, color=color, thickness=1)
            image = cv2.arrowedLine(image, (pyo.item(), pxo.item()), (pyf.item(), pxf.item()), color=color, thickness=1, tipLength=0.5)

    likelihood_image = cv2.applyColorMap(likelihood_image.astype(np.uint8), cv2.COLORMAP_JET)
    return image, likelihood_image
