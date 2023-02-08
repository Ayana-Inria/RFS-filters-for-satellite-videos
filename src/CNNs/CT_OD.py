# import _init_paths
import os

import torch
import torch.utils.data
from src.CNNs import opts
from src.CNNs.model.model import create_model, load_model, save_model
import cv2
from src.CNNs.detector import Detector2 as Detector3
import numpy as np
import csv
from src.CNNs.CT_OD import opts
from src.CNNs.utils.image import draw_umich_gaussian
from os import listdir


# TEST: AOIs 02, 03, 34   TRAIN: 40, 41,42;
# TEST: AOIs 01, 40,      TRAIN: 34, 41,42;
# TEST: AOIs 04, 41,      TRAIN: 42, 34, 40


def generate_pre_heatma_from_cnn_detections(rows, cols, cnn_detections, score_thres=0.2):
    pre_hm = np.zeros((rows, cols))
    if 'results' in cnn_detections.keys():
        for bb in cnn_detections['results']:
            py, px = bb['ct'].astype(np.int32)
            score = bb['score']
            if score > score_thres:
                draw_umich_gaussian(pre_hm, [py, px], 10, k=1.0)
    pre_hm = torch.from_numpy(pre_hm).unsqueeze(0).unsqueeze(0).float()
    return pre_hm

def highestmultiple(n, k=32):
    """
    Returns highest multiple of k closest to n
    :param n:
    :param k:
    :return:
    """
    answer = n
    while(answer % k):
        answer -= 1
    return answer



def create_detector(parameters, rows, cols):
    opt = opts.opts().parse()
    opt.num_classes = 1
    opt.pre_hm = True
    opt.input_h = rows  # highestmultiple(sample_img.shape[0], k=32)  #576  # 579
    opt.input_w = cols  # highestmultiple(sample_img.shape[1], k=32) #544  # 558
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    model = create_model(arch='generic', head={'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2}, # ,, 'ltrb_amodal': 4},
                      head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]},
                       opt=opt) #, 'ltrb_amodal': [256]},)
    model = load_model(model, parameters['dictionary_path'], opt)
    detector = Detector3(opt, model)
    return detector

def test_only(opt, sample_img, testing_output_path):

    directories = ['BBs', 'heatmap', 'heatmap_only', 'heatmap_gt', 'images', 'regression_only', 'labels']
    for temp_dir in directories:
        curr_dir = os.path.join(testing_output_path, temp_dir)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

    X = []
    for idx in range(len(batches)):
        print(counter)
        batch = batches[idx]
        img = batch['image'].transpose(1, 2, 0)
        pre_hm = None
        if opt.pre_hm:
            if idx > 0:
                pre_hm = batches[idx - 1]['hm']
            else:
                pre_hm = batches[idx]['hm']
            pre_hm = cv2.resize(pre_hm[0, ...], (img.shape[1], img.shape[0]))
            pre_hm = torch.from_numpy(pre_hm).unsqueeze(0).unsqueeze(0)

            hm_gt = batches[idx]['hm'] * 255
            hm_gt = cv2.resize(hm_gt[0, ...], (img.shape[1], img.shape[0]))
            hm_gt = hm_gt.astype(np.uint8)

        ret = detector.run(img, pre_hm)
        # meta['pre_dets'] = ret['results']
        # OG IMAGE
        img = (img * 255).astype(np.uint8)
        img_labels = (img[:, :, 0] * 0).astype(np.uint16)
        rows, cols, chas = img.shape
        cv2.imwrite(os.path.join(testing_output_path, 'images','og_image_' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), img)

        # HEATMAP
        hm = (ret['output']['hm'][0, 0, ...].cpu().numpy() * 255).astype(np.uint8)
        hm = cv2.resize(hm, (cols, rows))
        hm_save = cv2.addWeighted(hm, 0.5, img[:, :, 0], 0.7, 0)
        cv2.imwrite(os.path.join(testing_output_path, 'heatmap', 'heatmap' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), hm_save)
        cv2.imwrite(os.path.join(testing_output_path, 'heatmap_only', 'heatmap' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), hm)

        gt_eval1 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        gt_eval2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        gt_eval3 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        gt_eval1[:, :, 1] += hm
        gt_eval2[:, :, 2] += hm_gt
        overlap = hm * hm_gt
        idx = np.where(overlap > 0)
        hm[idx] = 0
        hm_gt[idx] = 0
        gt_eval3[:, :, 0] = (overlap).astype(np.uint8)
        gt_eval3[:, :, 1] = hm
        gt_eval3[:, :, 2] = hm_gt
        gt_eval3 += img
        gt_overlap1 = img + gt_eval1
        gt_overlap2 = img + gt_eval2

        gt_eval = np.concatenate((gt_eval3, gt_overlap1, gt_overlap2), axis=1)
        cv2.imwrite(os.path.join(testing_output_path, 'heatmap_gt', 'heatmap_gt_' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), gt_eval)



        # BOUNDING BOXES
        img_bbs = img.copy()
        Xk = {}
        Xk[-999] = counter

        for bb in ret['results']:
            px, py = bb['ct'].astype(np.int32)
            vx, vy = bb['tracking']
            obj_id = bb['tracking_id']
            img_bbs = cv2.circle(img_bbs, (px.item(), py.item()), 3, (0, 0, 255), 1)
            img_labels[py, px] = obj_id
            Xk[obj_id] = np.array([py, px, vy, vx, 5, 5])
        X.append((Xk))

        print(img_labels.shape)
        cv2.imwrite(os.path.join(testing_output_path, 'BBs', 'BBs' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), img_bbs)
        cv2.imwrite(os.path.join(testing_detection_path, 'labels' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), img_labels)

        trajectories = plot_trajectories(X, img.copy())
        save_show_cv_image(trajectories, output_directory=testing_output_path + "/PARTICLES", data_name="trajectories", k=counter, show_image=False)
        # REGRESSION
        reg = ret['output']['tracking'][0, ...].cpu().numpy().transpose(1, 2, 0)
        reg = cv2.resize(reg, (cols, rows))
        reg = np.concatenate((reg, np.zeros((rows, cols, 1))), axis=2)
        reg_save = cv2.addWeighted(((reg - reg.min())/(reg.max() - reg.min()) * 255).astype(np.uint8), 0.5, img, 0.7, 0)
        # cv2.imwrite(os.path.join(testing_output_path, 'regression', 'regression' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), reg_save)

        img_vels = img.copy()
        Zk = []
        for bb in ret['results']:
            py, px = bb['ct'].astype(np.int32)
            try:
              vy, vx = reg[px, py, 0:2].astype(np.float32)
            except IndexError:
              vx, vy = 0, 0
              # print("Errors w velocity")
            Zk.append([px, py, -vx, -vy, 5, 5])

        Zk = np.array(Zk)
        Zk = Zk.transpose(1, 0)
        img_vels = plot_measurements_w_velocity(Zk, img_vels)
        cv2.imwrite(os.path.join(testing_output_path, 'regression_only', 'regression' + str(counter).zfill(4) + '_' + str(epoch) + '_' + '.png'), img_vels)
        if(counter % 25 == 0):
            save_object_states(X, testing_output_path, partial_k=counter)
        counter += 1
    save_object_states(X, testing_output_path)


def plot_measurements_w_velocity(Zk, image_og, L=4):
    image = np.copy(image_og)
    if len(Zk.shape) < 2:
        return image
    colors = (255 * np.random.rand(Zk.shape[1], 3)).astype(np.uint8)
    # print(colors[:, 0])
    for i in range(Zk.shape[1]):
        color = (255, 255, 255) # tuple(colors[i, j].item() for j in range(3))
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



def plot_trajectories(X, image, thickness=2):
    np.random.seed(0)
    colors = (255 * np.random.rand(200, 3)).astype(np.uint8)
    i = 0
    num_frames = len(X)
    for label in X[-1].keys():
        if label < 0:
            continue
        x_est = X[num_frames - 1][label]
        pix = x_est[0].round().astype(np.int32)
        piy = x_est[1].round().astype(np.int32)
        l_idx = (label - 1) % 200
        color = tuple(colors[l_idx, j].item() for j in range(3))
        image = cv2.circle(image, (piy, pix), radius=1, color=color, thickness=thickness)
        i = i + 1

        current_frame = num_frames - 2
        temp_counter = 0
        while current_frame >= 0:
            # start from the second last
            if label in X[current_frame].keys():
                x_prev = X[current_frame][label]
                pfx = x_prev[0].round().astype(np.int32)
                pfy = x_prev[1].round().astype(np.int32)
                image = cv2.line(image,  (piy, pix), (pfy, pfx), color, thickness=thickness)
                # image = cv2.circle(image, (piy, pix), radius=1, color=color, thickness=1)
                piy = pfy
                pix = pfx
            current_frame -= 1
            if temp_counter > 5:
                break
            else:
                temp_counter += 1
    return image


def save_show_cv_image(image, output_directory=".", data_name="particles", k=0, show_image=False):
    if show_image:
        cv2.imshow(data_name, image)
        cv2.waitKey(0)
    write_dir = os.path.join(output_directory, data_name)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    cv2.imwrite(os.path.join(write_dir, f'{data_name}_{k:04}.png'), image)


def save_object_states(object_states, directory, FPS=None, partial_k=None):
    '''
        INPUTS:
            object_states: list of dictionaries
                            each dictionary key: object label
                                              value: np array [1, 6] for object state
        OUTPUTS:
            csv file with this information
            values of -9999999 are added where the object does not exists

    '''
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
    if(partial_k):
        name = 'object_states_{}.csv'.format(partial_k)
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


if __name__ == '__main__':
  opt = opts().parse()
  # main(opt)
  test_only(opt)
