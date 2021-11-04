import cv2
import os
import sys
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
from PIL import Image
import pickle

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize


from iou_utils import get_max_iou, get_good_iou, get_max_iou2, get_om_iou, get_max_iou3, get_max_iouo, average_precision_score2

def compute_iou(a, b):
    b= list(map(int,b))
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    # print('inter', inter)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    # print('uni', union)
    return inter * 1.0 / union

def set_box_pid(boxes, box, pids, pid):
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    print("Person: %s, box: %s cannot find in images." % (pid, box))

def image_path_at(data_path, image_index, i):
    image_path = osp.join(data_path, image_index[i])
    assert osp.isfile(image_path), "Path does not exist: %s" % image_path
    return image_path

def load_image_index(root_dir, db_name):
    """Load the image indexes for training / testing."""
    # Test images
    test = loadmat(osp.join(root_dir, "annotation", "pool.mat"))
    test = test["pool"].squeeze()
    test = [str(a[0]) for a in test]
    if db_name == "psdb_test":
        return test

    # All images
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]

    # Training images = all images - test images
    train = list(set(all_imgs) - set(test))
    train.sort()
    return train

if __name__ == "__main__":
    db_name = "psdb_test"
    root_dir = '/home/cvlab3/Downloads/AlignPS/demo/anno/kist/'


    with open('/home/cvlab3/Downloads/AlignPS/demo/anno/kist/gallery_cam1.json', 'r') as fid:
        test_det = json.load(fid)

    id_to_img = dict()
    img_to_id = dict()

    gallery_num= 0
    for td in test_det['images']:
        im_name = td['file_name'].split('/')[-1]
        im_id = td['id']
        id_to_img[im_id] = im_name
        img_to_id[im_name] = im_id
        gallery_num += 1


    with open('/home/cvlab3/Downloads/AlignPS/demo/anno/kist/query_cam1.json', 'r') as fid:
        test_det2 = json.load(fid)

    id_to_img2 = dict()
    img_to_id2 = dict()

    query_num= 0
    for td in test_det2['images']:
        # print(td)
        im_name2 = td['file_name'].split('/')[-1]
        im_id2 = td['id']
        id_to_img2[im_id2] = im_name2
        img_to_id2[im_name2] = im_id2
        query_num += 1

    results_path = '/home/cvlab3/Downloads/AlignPS/work_dirs/faster_rcnn_r50_caffe_c4_1x_cuhk_single_two_stage17_6_nae1/'

    x1 = '' #_x1
    with open(os.path.join(results_path, 'results_1000_gallery_cam1.pkl'), 'rb') as fid:
        all_dets = pickle.load(fid)


    with open(os.path.join(results_path, 'results_1000_query_cam1.pkl'), 'rb') as fid:
        all_dets2 = pickle.load(fid)

    gallery_dicts1 = {}
    gallery_dicts2 = {}
    all_dets1 = all_dets[0]

    # num = 0
    for i, dets in enumerate(all_dets1):

        if i >= 125:
            image_id = i + 1

        else:
            image_id = i

        gallery_dicts1[image_id] = dict()
        gallery_dicts1[image_id]['bbox'] = dets[0][:, :4]
        gallery_dicts1[image_id]['scores'] = dets[0][:, 4]
        gallery_dicts1[image_id]['feats'] = dets[0][:, 5:]
        gallery_dicts1[image_id]['pred_bbox'] = dets[0][:, :4]


        # img = cv2.imread('/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_new{}/'.format(x1) + id_to_img[image_id])
        r, = dets[0][:, 4].shape
        list_num = []
        for k in range(r):
            if gallery_dicts1[image_id]['scores'][k] > 0.2:
                numbers = gallery_dicts1[image_id]['bbox'][k]
                list_num.append(numbers)
                #here

                # l = int(numbers[0])
                # t = int(numbers[1])
                # r = int(numbers[2])
                # b = int(numbers[3])
                #
                # cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 3)

                # num = num + 1
        gallery_dicts1[image_id]['pred_bbox'] = np.array(list_num, dtype=object)
        # gallery_dicts1[image_id]['pred_feat'] = gallery_dicts1[image_id]['feats']
        #cv2.imwrite("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_gallery_cam1{}_result/{}_{}.jpeg".format(x1, i, num), img)  # save img



    ##########
    #query
    query_dicts1 = {}
    all_dets2 = all_dets2[0]
    num2 = 0

    for i, dets in enumerate(all_dets2):
        image_id = i
        query_dicts1[image_id] = dict()
        query_dicts1[image_id]['bbox'] = dets[0][:, :4]
        query_dicts1[image_id]['scores'] = dets[0][:, 4]
        query_dicts1[image_id]['feats'] = dets[0][:, 5:]
        query_dicts1[image_id]['pred_bbox'] = dets[0][:, :4]

        # img2 = cv2.imread('/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_new{}/'.format(x1) + id_to_img2[image_id])

        r, = dets[0][:, 4].shape
        list_num = []
        for k in range(r):
            if query_dicts1[image_id]['scores'][k] > 0.2:
                # print('bbox',gallery_dicts1[image_id]['bbox'][i])

                numbers = query_dicts1[image_id]['bbox'][k]
                # print(numbers)
                list_num.append(numbers)

                # l = int(numbers[0])
                # t = int(numbers[1])
                # r = int(numbers[2])
                # b = int(numbers[3])
                # # print((l,r), (r,b))
                # cv2.rectangle(img2, (l, t), (r, b), (0, 0, 255), 3)

                num2 = num2 + 1
        query_dicts1[image_id]['pred_bbox'] = np.array(list_num, dtype=object)
        # query_dicts1[image_id]['pred_feat'] = query_dicts1[image_id]['feats']
       # cv2.imwrite("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_query_cam1{}_result/{}_{}.jpeg".format(x1, image_id, num2),img2)  # save img

    thres = 0.2
    ap = 0
    precision = {}
    recall = {}

    aps = []
    accs = []
    topk = [1,5,10]


    gallery_gt_box = [b['bbox'] for b in test_det['annotations']]
    # gallery_scores = [b['scores'] for b in test_det['annotations']]
    # gallery_boxes =
    # gallery_feats = [b[0][:, 5:] for b in all_dets1 if b[0][:, 4] > 0.2]

    for image_id in range(query_num):
        # if image_id == 125:
        #     continue

        count_gt, count_tp = 0, 0
        query_boxes = query_dicts1[image_id]['pred_bbox']  # predicted bb
        query_boxes = np.array([list(map(int, q)) for q in query_boxes])
        query_feats = query_dicts1[image_id]['feats']
        # muti
        query_gt_box = np.array([b['bbox'] for b in test_det2['annotations'] if b['image_id']==image_id])
        # single
        # query_gt_box =test_det2['annotations'][image_id]['bbox']
        if len(query_boxes) ==0:
            continue
        for qgb in query_gt_box:
            iou, iou_max, nmax = get_max_iouo(query_boxes, query_gt_box)
            query_feat = query_feats[nmax]
            query_feat = normalize(query_feat[np.newaxis, :], axis=1).ravel()
            for i in range(gallery_num):
                if i == 125:
                    continue
                y_true, y_score = [], []
                gallery_gt_box = [b['bbox'] for b in test_det['annotations'] if b['image_id'] == i]
                # gallery_gt_box = test_det['annotations'][i]['bbox']
                count_gt = len(gallery_gt_box)
                # img_gid = i
                det = np.asarray(gallery_dicts1[i]['bbox'])
                scores = np.asarray(gallery_dicts1[i]['scores'])
                keep_inds = np.where(scores >= thres)
                scores = scores[keep_inds]
                det = det[keep_inds]
                gallery_feat = gallery_dicts1[i]['feats'][keep_inds]
                if gallery_feat.shape[0] > 0:
                    gallery_feat = normalize(gallery_feat, axis=1)
                else:
                    continue

                sim = gallery_feat.dot(query_feat).ravel()
                # print('sim', sim)
                label = np.zeros(len(sim), dtype=np.int32)
                # m = 0
                # print('gt', count_gt)
                if count_gt > 0:
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    for j, roi in enumerate(det[:, :4]):
                        for ggb in gallery_gt_box:
                            w, h = int(ggb[2]), int(ggb[3])
                            iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                            if compute_iou(roi, ggb) >= iou_thresh:
                                label[j] = 1
                                count_tp += 1
                                break

                # print(i, type(y_true))
                y_true.extend(list(label))
                y_score.extend(list(sim))

                y_score = np.asarray(y_score)
                y_true = np.asarray(y_true)
                # print('ys',y_score)
            # assert count_tp <= count_gt
                if count_tp <= count_gt or count_gt == 0: continue
                recall_rate = count_tp * 1.0 / count_gt
                ap = 0 if count_tp == 0 else average_precision_score2(y_true, y_score) * recall_rate
                print('ap', ap)
                aps.append(ap)
            # inds = np.argsort(y_score)[::-1]
            # y_score = y_score[inds]
            # y_true = y_true[inds]
            # accs.append([min(1, sum(y_true[:k])) for k in topk])

    print("threshold: ", thres)
    print("  mAP = {:.2%}".format(np.mean(aps)))
    # accs = np.mean(accs, axis=0)
    # for i, k in enumerate(topk):
    #     print("  Top-{:2d} = {:.2%}".format(k, accs[i]))

    # all_thresh = [0.2, 0.2, 0.2]
