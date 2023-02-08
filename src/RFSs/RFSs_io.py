import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from scipy.optimize import linear_sum_assignment
import src.RFSs.TBD_glmb.tbd_helper_functions as tbd_fns
import os
import csv
import cv2

from src.CNNs.utils.image import draw_umich_gaussian

def get_Zk(data_at_time_k, specific_labels=None):
    '''
        Returns np array of shape [num_measurements x 2] for center coords of each measurement
    '''
    Zk = []
    for label in np.unique(data_at_time_k):
        if label == 0:
            continue
        if specific_labels is not None and label not in specific_labels:
            continue
        coords = np.where(data_at_time_k == label)
        Zk.append([coords[0].mean(), coords[1].mean(), 5, 5])
    Zk = np.array(Zk)
    return Zk

def get_Zk_and_Gt(data_at_time_k, specific_labels=None):
    Zk = []
    X_gt = {}
    for label in np.unique(data_at_time_k):
        if label == 0:
            continue
        if specific_labels is not None and label not in specific_labels:
            continue
        coords = np.where(data_at_time_k == label)
        x = coords[0].mean()
        y = coords[1].mean()
        Zk.append([x, y, 5, 5])
        X_gt[label] = [x, y, 0, 0, 5, 5]

    Zk = np.array(Zk)
    return Zk, X_gt

def generate_fake_energy_image_old(image, zk, R=2):
    for z in zk:
        x, y = z[0:2].astype(np.uint16)
        for ii in range(-R, R):
            for jj in range(-R, R):
                xx = max(min(image.shape[0] - 1, x.item() + ii), 0)
                yy = max(min(image.shape[1] - 1, y.item() + jj), 0)
                image[xx, yy] = 125.0
    image = gaussian_filter(image, 3)
    return image

#
def generate_fake_energy_image(image, zk, R=2):
    for z in zk:
        x, y = z[0:2].astype(np.uint16)
        draw_umich_gaussian(image, z[0:2].astype(np.uint16), radius=2, k=1.0)
    return image

def get_birth_particles_smc_w_velocities(Z,tt_pred, current_label=1, k=0, distance_thres=100, N_particles_per_birth_object=20, device=None):
    if device is None:
            device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

    # X_pred = np.array(X_pred)

    absolute_indexes = []

    num_components = tt_pred.Num_Ts
    X_pred = torch.zeros((tt_pred.Num_Ts, 6), device=device)
    for i in range(num_components):
        X_pred[i, :] = (tt_pred.Xs[i, :, :] * tt_pred.ws[i, ...]).sum(dim=0)

    if(len(X_pred.shape) > 0 and Z.shape[0] > 0):
        # Do measurement - component association
        cost = torch.cdist(Z[:, 0:2], X_pred[:, 0:2], p=2)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        distances = cost[row_ind, col_ind]
        reduced_indexes = (distances < distance_thres).nonzero()[:, 0]
        absolute_indexes = row_ind[reduced_indexes.cpu().numpy()]

    label = current_label
    H_birth = 0
    new_mu = []
    new_w = []

    L_birth = []

    for i in range(len(Z)):
        if i in absolute_indexes:
            continue

        z = Z[i, :]
        new_z = torch.tensor([z[0].cpu().item(), z[1].cpu().item(), z[2].cpu(), z[3].cpu(), 5, 5]).unsqueeze(0)
        partickes_for_z = generate_N_random_particles(mu=new_z, std=[2.0, 2.0, 0.1, 0.1,  0.0, 0.0], N_particles=N_particles_per_birth_object)
        new_w.append((torch.ones(N_particles_per_birth_object) / float(N_particles_per_birth_object)).unsqueeze(-1).to(device))

        new_mu.append(partickes_for_z)
        L_birth.append([k, label])
        label += 1
        H_birth += 1

    max_label = label
    if len(new_w):
        w_birth = torch.stack(new_w)
        Xs_birth = torch.stack(new_mu).to(device)
        num_components, num_particles, num_dims = Xs_birth.shape
        r_birth = torch.ones((num_components, 1), device=Xs_birth.device) * 0.9 # hardcoded r_birth
    else:
        w_birth = torch.tensor([], device=device)
        Xs_birth = torch.tensor([], device=device)
        r_birth = torch.tensor([], device=device)

    if len(new_mu) == 0:
        Xs_birth = Xs_birth.unsqueeze(-1)

    tt_birth = tbd_fns.Track_List_SMC()
    tt_birth.ws = w_birth
    tt_birth.Xs = Xs_birth
    tt_birth.Ls = L_birth
    tt_birth.r = r_birth
    tt_birth.Es = [None] * H_birth
    tt_birth.Num_Ts = H_birth

    if Z.shape[0] > 0:
        surviving_measurements = Z[absolute_indexes, :]
    else:
        surviving_measurements = torch.tensor([], device=Xs_birth.device)

    return tt_birth, surviving_measurements, max_label


def generate_N_random_particles(mu, std=[2.0, 2.0, 1.0, 1.0,0.0, 0.0], N_particles=20):
    '''
        Generate N random particles at this location
        mu: [1, 6]
        std: list of length(6)

    '''
    torch.manual_seed(0)
    means = torch.zeros((N_particles, 6))
    std = torch.tensor(std)
    random_perturbations = torch.normal(means, std)
    return mu + random_perturbations


def save_object_states(object_states, directory, FPS=None, object_detector=False):
    """
        INPUTS:
            object_states: list of dictionaries
                            each dictionary key: object label
                                              value: np array [1, 6] for object state
        OUTPUTS:
            csv file with this information
            values of -9999999 are added where the object does not exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    number_of_frames = len(object_states)
    # Find number of objects. Could be optimized
    map_of_labels = {}
    label_n = 1
    for frame in object_states:
        for k, v in frame.items():
            if(k not in map_of_labels.keys() and k != -999):
                map_of_labels[k] = label_n
                label_n += 1

    number_of_objects = label_n - 1

    # Create GT array. 6 values per  object (state vector). Allow + 1 col for frame number
    gt_values = np.zeros((number_of_frames, 6 * number_of_objects + 1)) - 9999999

    # Populate gt_values
    frame_counter = 1
    for frame in object_states:
        # Row number
        gt_values[frame_counter - 1, 0] = frame[-999]
        for label, x_v in frame.items():
            if(label != -999):
                # Store the values at each time step. Store 6 values for each object
                gt_values[frame_counter - 1, (map_of_labels[label] - 1) * 6 + 1: (map_of_labels[label] - 1) * 6 + 1 + 6] = x_v

        frame_counter += 1

    # Start creating CSV
    if object_detector:
        name = 'object_states_OD.csv'
    else:
        name = 'object_states.csv'
    with open(directory + '/' + name, mode='w', newline='') as state_file:
        csv_writer = csv.writer(state_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Create CSV file header
        header_content = []
        if(FPS is not None):
            header_content.append('FPS {:.2f}'.format(FPS))
        else:
            header_content.append('Frame Number')
        for object_n in map_of_labels.keys():
            header_content.append('obj_{}_px'.format(object_n))
            header_content.append('obj_{}_py'.format(object_n))
            header_content.append('obj_{}_vx'.format(object_n))
            header_content.append('obj_{}_vy'.format(object_n))
            header_content.append('obj_{}_ax'.format(object_n))
            header_content.append('obj_{}_ay'.format(object_n))
        csv_writer.writerow(header_content)

        for frame_n in range(gt_values.shape[0]):
            csv_writer.writerow(gt_values[frame_n, :])

def plot_debug(extras_dictionary, output_directory, k):
    # DEBUGGGG
    resampled = extras_dictionary['resampled_image']
    likelihood = extras_dictionary['updated_likelihood_image']

    # Birth
    birth_image = extras_dictionary['birth_image']

    # Surviving
    surviving_image1 = extras_dictionary['surviving_image1']
    surviving_image2 = extras_dictionary['surviving_image2']
    surviving_image3 = extras_dictionary['surviving_image3']
    surviving_image4 = extras_dictionary['surviving_image4']
    birth_particles = extras_dictionary['birth_image']
    X = extras_dictionary['X']
    Xk = X[-1]
    X_gt = extras_dictionary['X_gt']
    zk = extras_dictionary['zk']
    image_k = extras_dictionary['image_k']
    energy_image_k = extras_dictionary['energy_image_k']


    birth_particles_and_measurements = plot_measurements(np.transpose(zk.numpy()), birth_particles, L=8)
    surviving_image1_and_measurements = plot_measurements(np.transpose(zk.numpy()), surviving_image1, L=12)
    surviving_image2_and_measurements = plot_measurements(np.transpose(zk.numpy()), surviving_image2, L=12)
    surviving_image3_and_measurements = plot_measurements(np.transpose(zk.numpy()), surviving_image3, L=12)
    surviving_image4_and_measurements = plot_measurements(np.transpose(zk.numpy()), surviving_image4, L=12)

    # States
    states = plot_states(Xk, np.copy(image_k), R=8, L1=8, L2=20)
    states_measurements = plot_measurements(np.transpose(zk.numpy()), states, L=12)

    # Measurements
    measurements_og = plot_measurements_w_velocity(np.transpose(zk.numpy()), image_k, L=12)
    resampled_and_measurements = plot_measurements(np.transpose(zk.numpy()), resampled, L=12)

    # Trajectories + GT Trajectories
    trajectories = plot_trajectories(X, np.copy(image_k))
    trajectories = plot_measurements(np.transpose(zk.numpy()), trajectories)
    gt_trajectories = plot_trajectories(X_gt, np.copy(image_k))

    # Large Image 1
    super_large_image = np.concatenate((measurements_og,
                                        birth_image,
                                        gt_trajectories,
                                        trajectories,
                                        states_measurements), axis=1)
    # Large Image 2
    large_im2 = np.concatenate((birth_particles_and_measurements,
                                surviving_image1_and_measurements,
                                surviving_image2_and_measurements,
                                surviving_image3_and_measurements,
                                # surviving_image4_and_measurements,
                                resampled_and_measurements), axis=1)
    super_large_image = np.concatenate((super_large_image, large_im2), axis=0)
    super_large_image = cv2.resize(super_large_image, (super_large_image.shape[1] * 2, super_large_image.shape[0] * 2))

    # super_large_image = np.concatenate((large_im1, large_im2), axis=0)
    save_show_cv_image(super_large_image, output_directory=output_directory, data_name="DEBUG", k=k + 1,  show_image=False)
    save_show_cv_image(states, output_directory=output_directory, data_name="states", k=k + 1)
    save_show_cv_image(trajectories, output_directory=output_directory, data_name="trajectories", k=k + 1)
    save_show_cv_image(gt_trajectories, output_directory=output_directory, data_name="gt_trajectories", k=k + 1)
    save_show_cv_image(image_k, output_directory=output_directory, data_name="og_image", k=k + 1)

    likelihood = ((energy_image_k / 255.0)**2 *255)
    likelihood = cv2.applyColorMap(likelihood.astype(np.uint8), cv2.COLORMAP_JET)
    save_show_cv_image(likelihood, output_directory=output_directory, data_name="og_likelihood", k=k + 1)


def plot_measurements(Zk, image_og, L=4, color=(255,255,255)):
    image = np.copy(image_og)
    if len(Zk.shape) < 2:
        return image
    colors = (255 * np.random.rand(Zk.shape[1], 3)).astype(np.uint8)
    # print(colors[:, 0])
    for i in range(Zk.shape[1]):
        # color = (255, 255, 255) # tuple(colors[i, j].item() for j in range(3))
        pix = Zk[0, i].round().astype(np.int32)
        piy = Zk[1, i].round().astype(np.int32)
        image = cv2.rectangle(image, (piy - L, pix - L), (piy + L, pix + L), color=color, thickness=1)
    return image

def plot_measurements_w_velocity(Zk, image_og, L=4, color = (255, 255, 255)):
    image = np.copy(image_og)
    if len(Zk.shape) < 2:
        return image
    colors = (255 * np.random.rand(Zk.shape[1], 3)).astype(np.uint8)
    # print(colors[:, 0])
    for i in range(Zk.shape[1]):
        # tuple(colors[i, j].item() for j in range(3))
        pix = Zk[0, i].round().astype(np.int32)
        piy = Zk[1, i].round().astype(np.int32)

        v = np.array([Zk[2, i], Zk[3, i]])

        vel_dir1 = v // 8
        vel_dir2 = v

        pxo = (pix + vel_dir1[0]).round().astype(np.int32)
        pyo = (piy + vel_dir1[1]).round().astype(np.int32)

        pxf = (pix + vel_dir2[0]).round().astype(np.int32)
        pyf = (piy + vel_dir2[1]).round().astype(np.int32)

        image = cv2.rectangle(image, (piy - L, pix - L), (piy + L, pix + L), color=color, thickness=1)
        image = cv2.arrowedLine(image, (pyo, pxo), (pyf, pxf), color=color, thickness=1, tipLength=0.5)

    return image

def plot_states(x_est_dict, image, R=1, L1=2, L2=4, default_color=None):
    np.random.seed(0)
    colors = (255 * np.random.rand(200, 3)).astype(np.uint8)
    # print(colors[:, 0])
    i = 0
    for k in x_est_dict.keys():
        if(k < 0):
            continue
        x_est = x_est_dict[k].numpy()
        pix = x_est[0].round().astype(np.int32)
        piy = x_est[1].round().astype(np.int32)

        v = np.array([x_est[2], x_est[3]])
        phi = np.degrees(np.arctan2(v[1], v[0]) + np.pi / 2.0)
        if phi < 0:
            phi = phi + 180

        k_idx = (k - 1) % 200
        color = tuple(colors[k_idx, j].item() for j in range(3))

        vel_dir1 = v / np.linalg.norm(v) * L1
        vel_dir2 = v / np.linalg.norm(v) * L2

        pxo = (x_est[0] + vel_dir1[0]).round().astype(np.int32)
        pyo = (x_est[1] + vel_dir1[1]).round().astype(np.int32)

        pxf = (x_est[0] + vel_dir2[0]).round().astype(np.int32)
        pyf = (x_est[1] + vel_dir2[1]).round().astype(np.int32)

        image = cv2.circle(image, (piy, pix), radius=R, color=color, thickness=2)
        image = cv2.arrowedLine(image, (pyo, pxo), (pyf, pxf), color=color, thickness=2, tipLength=0.5)
        # image = cv2.putText(image, str(k), (pyo, pxo + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        i = i + 1
    return image

def plot_trajectories(X, image, thickness=2, trajectory_length=100000):
    np.random.seed(0)
    colors = (255 * np.random.rand(200, 3)).astype(np.uint8)
    i = 0
    num_frames = len(X)
    for label in X[-1].keys():
        if label < 0:
            continue
        x_est = X[num_frames - 1][label]
        pix = int(x_est[0].round().item())
        piy = int(x_est[1].round().item())
        l_idx = (label - 1) % 200
        if label < 10000:
            color = tuple(colors[l_idx, j].item() for j in range(3))
            radius = 1
        else:
            color = tuple((colors[l_idx, 0].item(), colors[l_idx, 1].item(), 255))
            radius = 2

        image = cv2.circle(image, (piy, pix), radius=radius, color=color, thickness=thickness)
        i = i + 1

        # start from the second last
        previous_frame = num_frames - 2
        length_counter = 0
        while previous_frame >= 0 and length_counter <= trajectory_length:
            if label in X[previous_frame].keys():
                x_prev = X[previous_frame][label]
                pfx = int(x_prev[0].round().item())
                pfy = int(x_prev[1].round().item())
                image = cv2.line(image,  (piy, pix), (pfy, pfx), color, thickness=thickness)
                # image = cv2.circle(image, (piy, pix), radius=1, color=color, thickness=1)
                piy = pfy
                pix = pfx
            previous_frame -= 1
            length_counter += 1
    return image



def plot_trajectories_w_velocities(X, image, thickness=2, trajectory_length=100000, add_text=False, specific_labels=None):
    np.random.seed(0)
    colors = (255 * np.random.rand(200, 3)).astype(np.uint8)
    i = 0
    num_frames = len(X)
    for label in X[-1].keys():
        if label < 0:
            continue
        if specific_labels:
            if label not in specific_labels:
                continue
        x_est = X[num_frames - 1][label]
        pix = int(x_est[0].round().item())
        piy = int(x_est[1].round().item())
        l_idx = (label - 1) % 200
        if label < 10000:
            color = tuple(colors[l_idx, j].item() for j in range(3))
            radius = 1
        else:
            color = tuple((colors[l_idx, 0].item(), colors[l_idx, 1].item(), 255))
            radius = 2

        image = cv2.circle(image, (piy, pix), radius=radius + 1, color=color, thickness=thickness)

        v = np.array([x_est[2], x_est[3]])

        vel_dir1 = v // 8
        vel_dir2 = v

        pxo = (pix + vel_dir1[0]).round().astype(np.int32)
        pyo = (piy + vel_dir1[1]).round().astype(np.int32)

        pxf = (pix + vel_dir2[0]).round().astype(np.int32)
        pyf = (piy + vel_dir2[1]).round().astype(np.int32)
        image = cv2.arrowedLine(image, (pyo, pxo), (pyf, pxf), color=color, thickness=1, tipLength=0.5)

        if add_text:
            image = cv2.putText(image, str(label),  (pyo, pxo), cv2.FONT_HERSHEY_SIMPLEX,  0.2, color, 1, cv2.LINE_AA)

        i = i + 1

        # start from the second last
        previous_frame = num_frames - 2
        length_counter = 0
        while previous_frame >= 0 and length_counter <= trajectory_length:
            if label in X[previous_frame].keys():
                x_prev = X[previous_frame][label]
                pfx = int(x_prev[0].round().item())
                pfy = int(x_prev[1].round().item())
                image = cv2.line(image,  (piy, pix), (pfy, pfx), color, thickness=thickness)

                # image = cv2.circle(image, (piy, pix), radius=1, color=color, thickness=1)
                piy = pfy
                pix = pfx
            previous_frame -= 1
            length_counter += 1
    return image



def save_show_cv_image(image, data_name="particles", k=0, output_directory=None, show_image=False):
    if show_image:
        cv2.imshow(data_name, image)
        cv2.waitKey(0)
    if output_directory is not None:
        write_dir = os.path.join(output_directory, data_name)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        cv2.imwrite(os.path.join(write_dir, f'{data_name}_{k:04}.png'), image)

def update_vector_field(vector_field, X, k):
    Xk = X[k]
    for label in Xk.keys():
        if label < 0:
            continue
        px, py, vx, vy, w, h = torch.from_numpy(Xk[label]).to(vector_field.device)
        for ii in range(-2, 2):
            for jj in range(-2, 2):
                ci = min(max(int(ii + px), 0), vector_field.shape[0] - 1)
                cj = min(max(int(jj + py), 0), vector_field.shape[1] - 1)
                if vector_field[ci, cj, 0] > 0:
                    vector_field[ci, cj, 0] = (vx + vector_field[ci, cj, 0]) / 2.0
                else:
                    vector_field[ci, cj, 0] = vx

                if vector_field[ci, cj, 1] > 0:
                    vector_field[ci, cj, 1] = (vy + vector_field[ci, cj, 1]) / 2.0
                else:
                    vector_field[ci, cj, 1] = vy

        if k > 0 and label in X[k - 1].keys():
            px_prev, py_prev, vx_prev, vy_prev, w, h = torch.from_numpy(X[k-1][label]).to(vector_field.device)
            points = intermediates([px_prev, py_prev], [px, py])
            for pii, pjj in points:
                for ii in range(-2, 2):
                    for jj in range(-2, 2):
                        ci = min(max(int(ii + pii), 0), vector_field.shape[0] - 1)
                        cj = min(max(int(jj + pjj), 0), vector_field.shape[1] - 1)
                        if(vector_field[ci, cj, 0] > 0):
                            vector_field[ci, cj, 0] = (vx + vector_field[ci, cj, 0]) / 2.0
                        else:
                            vector_field[ci, cj, 0] = vx

                        if(vector_field[ci, cj, 1] > 0):
                            vector_field[ci, cj, 1] = (vy + vector_field[ci, cj, 0]) / 2.0
                        else:
                            vector_field[ci, cj, 1] = vy

    return vector_field


def intermediates(p1, p2, nb_points=5):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)]
            for i in range(1, nb_points+1)]

def birth_field_plotter(birth_field, image_k):
    birth_mins = birth_field.min()
    birth_maxs = birth_field.max()
    idxs = (birth_field == np.array([0, 0, 0]))
    birth_field_to_plot = ((birth_field - birth_mins) / (birth_maxs - birth_mins) * 255).numpy().astype(np.uint8)
    birth_field_to_plot[idxs] = 0
    return cv2.addWeighted(birth_field_to_plot, 0.8, image_k, 0.7, 0)


def highestmultiple(n, k=32):
    answer = n
    while(answer % k):
        answer -= 1
    return answer

'''


def get_Zk_and_Gt_w_vel(data_at_time_k, data_at_time_kp1, specific_labels=None):
    Zk = []
    X_gt = {}

    labels_at_k = np.unique(data_at_time_k)
    labels_at_kp1 = np.unique(data_at_time_kp1)
    for label in labels_at_k:
        if label == 0:
            continue
        if specific_labels is not None and label not in specific_labels:
            continue
        coords = np.where(data_at_time_k == label)
        x = coords[0].mean()
        y = coords[1].mean()

        # Get velocity
        if label in labels_at_kp1:
            coords = np.where(data_at_time_kp1 == label)
            xp1 = coords[0].mean()
            yp1 = coords[1].mean()
            vx = xp1 - x
            vy = yp1 - y
        else:
            vx = 0
            vy = 0

        Zk.append([x, y, vx, vy, 5, 5])
        X_gt[label] = [x, y, vx, vy, 5, 5]

    Zk = np.array(Zk)
    return Zk, X_gt
'''
