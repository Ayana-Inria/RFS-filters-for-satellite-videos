from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import time
import torch
import math

from src.CNNs.model.decode import generic_decode
from src.CNNs.model.utils import flip_tensor, flip_lr_off, flip_lr
from src.CNNs.utils.image import get_affine_transform, affine_transform, draw_umich_gaussian
from src.CNNs.utils.tracker import Tracker
from src.CNNs.utils.post_process import generic_post_process
from src.CNNs.dataset.custom_dataset import CustomDataset


class Detector2(object):
  def __init__(self, opt, model):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    print('Creating model...')
    self.model = model.to(opt.device)
    self.model.eval()

    self.opt = opt
    self.trained_dataset = CustomDataset
    self.mean = np.array(
      self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(
      self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
    self.pause = not opt.no_pause
    self.rest_focal_length = self.trained_dataset.rest_focal_length if self.opt.test_focal_length < 0 else self.opt.test_focal_length
    self.flip_idx = self.trained_dataset.flip_idx
    self.cnt = 0
    self.pre_images = None
    self.pre_image_ori = None
    self.tracker = Tracker(opt)



  def run(self, images, pre_hm=None, pre_ind=None, meta={}):
    # run 2
    detections = []
    images = torch.from_numpy(images)

    scale = 1
    images, meta = self.pre_process(images.cpu().numpy(), scale, meta)
    images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
    # initializing tracker
    # pre_hms, pre_inds = None, None
    # initialize the first frame
    if self.pre_images is None:
        self.pre_images = images
        self.tracker.init_track(meta['pre_dets'] if 'pre_dets' in meta else [])
    if self.opt.pre_hm:
      pre_hm = pre_hm.to(self.opt.device, non_blocking=self.opt.non_block_test)
      # pre_hms2, pre_inds2 = self._get_additional_inputs(self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
      pre_hm = pre_hm[:, :, :self.opt.input_h, :self.opt.input_w]

    output, dets, forward_time = self.process(images.to(self.opt.device), self.pre_images.to(self.opt.device), pre_hm, pre_ind, return_time=True)

    # convert the cropped and 4x downsampled output coordinate system
    # back to the input image coordinate system
    result = self.post_process(dets, meta, scale)
    detections.append(result)

    # merge multi-scale testing results
    results = self.merge_outputs(detections)

    if self.opt.tracking:
      # public detection mode in MOT challenge
      public_det = None
      # add tracking id to results
      results = self.tracker.step(results, public_det)
      self.pre_images = images

    # Get measurements
    zk_4 = []
    zk_6 = []
    Xk = {}
    for bb in results:
        px, py = bb['ct']
        vx, vy = bb['tracking']
        obj_id = bb['tracking_id']
        Xk[obj_id] = [py, px, -vy, -vx, 5, 5]

        zk_4.append([py, px, 5.0, 5.0])
        zk_6.append([py, px, -vy, -vx, 5, 5])

    zk_4 = torch.tensor(zk_4, device=self.opt.device)  # zk_4 is (n_meas, 4)
    zk_6 = torch.tensor(zk_6, device=self.opt.device)  # zk_6 is (n_meas, 6)

    ret = {'results': results, 'output': output, 'zk_4': zk_4, 'Xk': Xk, 'zk_6': zk_6}
    return ret


  def pre_process(self, image, scale, input_meta={}):
    '''
    pre_process2
    Crop, resize, and normalize image. Gather meta data for post processing
      and tracking.
    '''
    resized_image, c, s, inp_width, inp_height, height, width = \
      self._transform_scale(image)
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, meta



  def _trans_bbox(self, bbox, trans, width, height):
    '''
    Transform bounding boxes according to image crop.
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


  def _transform_scale(self, image, scale=1):
    '''
      Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/
        keep original resolution but pad to a multiplication of 32
    '''
    image = image[:self.opt.input_h, :self.opt.input_w, :]
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_short > 0:
      if height < width:
        inp_height = self.opt.fix_short
        inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
      else:
        inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
        inp_width = self.opt.fix_short
      c = np.array([width / 2, height / 2], dtype=np.float32)
      s = np.array([width, height], dtype=np.float32)
    elif self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
      # s = np.array([inp_width, inp_height], dtype=np.float32)
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, c, s, inp_width, inp_height, height, width


  def _get_additional_inputs(self, dets, meta, with_hm=True):
    '''
    Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []
    for det in dets:
      if det['score'] < self.opt.pre_thresh or det['active'] == 0:
        continue
      bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
      bbox_out = self._trans_bbox(
        det['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = 8 # gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if with_hm:
          draw_umich_gaussian(input_hm[0], ct_int, radius)
        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2,
           (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])
    if with_hm:
      input_hm = input_hm[np.newaxis]
      if self.opt.flip_test:
        input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return input_hm, output_inds


  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                        [0, self.rest_focal_length, height / 2, 0],
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    if 'hm_hp' in output:
      output['hm_hp'] = output['hm_hp'].sigmoid_()
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    return output


  def _flip_output(self, output):
    average_flips = ['hm', 'wh', 'dep', 'dim']
    neg_average_flips = ['amodel_offset']
    single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
      'hp_offset', 'rot', 'tracking', 'pre_hm']
    for head in output:
      if head in average_flips:
        output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
      if head in neg_average_flips:
        flipped_tensor = flip_tensor(output[head][1:2])
        flipped_tensor[:, 0::2] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2
      if head in single_flips:
        output[head] = output[head][0:1]
      if head == 'hps':
        output['hps'] = (output['hps'][0:1] +
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      if head == 'hm_hp':
        output['hm_hp'] = (output['hm_hp'][0:1] + \
          flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

    return output


  def process(self, images, pre_images=None, pre_hms=None,
    pre_inds=None, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      '''
      print(images.device)
      print(pre_hms)
      print(pre_images.device)
      print(self.model.device)
      '''
      self.model = self.model.to(self.opt.device)
      output = self.model(images, pre_images, pre_hms)[-1]
      output = self._sigmoid_output(output)
      output.update({'pre_inds': pre_inds})
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = generic_decode(output, K=self.opt.K, opt=self.opt)
      torch.cuda.synchronize()
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = generic_post_process(self.opt, dets, [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], threshold_detection=0.1)
    self.this_calib = meta['calib']

    return dets[0]

  def merge_outputs(self, detections):
    results = []
    for i in range(len(detections[0])):
      if detections[0][i]['score'] > 0:# self.opt.out_thresh:
        results.append(detections[0][i])
    return results

  def debug(self, debugger, images, dets, output, scale=1,
    pre_images=None, pre_hms=None):
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if 'hm_hp' in output:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')

    if pre_images is not None:
      pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
      pre_img = np.clip(((
        pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      debugger.add_img(pre_img, 'pre_img')
      if pre_hms is not None:
        pre_hm = debugger.gen_colormap(
          pre_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')


  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='generic')
    if self.opt.tracking:
      debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image,
        img_id='previous')
      self.pre_image_ori = image

    for j in range(len(results)):
      if results[j]['score'] > self.opt.vis_thresh:
        if 'active' in results[j] and results[j]['active'] == 0:
          continue
        item = results[j]
        if ('bbox' in item):
          sc = item['score'] if self.opt.demo == '' or \
            not ('tracking_id' in item) else item['tracking_id']
          sc = item['tracking_id'] if self.opt.show_track_color else sc

          debugger.add_coco_bbox(
            item['bbox'], item['class'] - 1, sc, img_id='generic')

        if 'tracking' in item:
          debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')

        tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
        if 'tracking_id' in item and self.opt.demo == '' and \
          not self.opt.show_track_color:
          debugger.add_tracking_id(
            item['ct'], item['tracking_id'], img_id='generic')

        if (item['class'] in [1, 2]) and 'hps' in item:
          debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
            img_id='generic')

    if len(results) > 0 and \
      'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
      debugger.add_3d_detection(
        image if not self.opt.qualitative else cv2.resize(
          debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])),
        False, results, self.this_calib,
        vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
      debugger.add_bird_view(
        results, vis_thresh=self.opt.vis_thresh,
        img_id='bird_pred', cnt=self.cnt)
      if self.opt.show_track_color and self.opt.debug == 4:
        del debugger.imgs['generic'], debugger.imgs['bird_pred']
    if 'ddd_pred' in debugger.imgs:
      debugger.imgs['generic'] = debugger.imgs['ddd_pred']
    if self.opt.debug == 4:
      debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
    else:
      debugger.show_all_imgs(pause=self.pause)


  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_image_ori = None

