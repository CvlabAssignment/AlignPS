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

from iou_utils import get_max_iou, get_good_iou

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
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
    # root_dir = '~/Downloads/WRCAN-PyTorch/src/image'

    # images_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_cam/'


    root_dir = '/home/cvlab3/Downloads/AlignPS/demo/anno/kist/'

    with open('/home/cvlab3/Downloads/AlignPS/demo/anno/kist/test_new.json', 'r') as fid:
        test_det = json.load(fid)




    id_to_img = dict()
    img_to_id = dict()

    img_num = 0
    for td in test_det['images']:
        im_name = td['file_name'].split('/')[-1]
        im_id = td['id']
        id_to_img[im_id] = im_name
        img_to_id[im_name] = im_id
        img_num += 1

    # print('0 img',id_to_img[0])
    # print(len(id_to_img))
    # image_file_names = os.listdir(images_path)
    # id_to_img = dict()
    # img_to_id = dict()
    #
    #
    # for idx, f in enumerate(image_file_names):
    #     _, frame_no = f.split("|")
    #     file_name = f
    #     id_to_img[idx] = file_name
    #     img_to_id[file_name] = idx
    # print('sys',sys.argv[1])
    results_path = '/home/cvlab3/Downloads/AlignPS/work_dirs/faster_rcnn_r50_caffe_c4_1x_cuhk_single_two_stage17_6_nae1/'# + sys.argv[1]
    #results_path = '/raid/yy1/mmdetection/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_cuhk_reid_1500_stage1_fpncat_dcn_epoch24_multiscale_focal_x4_bg-2_sub_triqueue_nta_nsa'
    #results_path = '/raid/yy1/mmdetection/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_cuhk_reid_1000_fpncat'
    # 'results_1000_x1'
    # x1 = '_x1' #'_x1' ''
    x1 = '_x1'
    with open(os.path.join(results_path, 'results_1000{}.pkl'.format(x1)), 'rb') as fid:
        all_dets = pickle.load(fid)

    # print(all_dets[0])
    gallery_dicts1 = {}
    # gallery_dicts2 = {}
    all_dets1 = all_dets[0]
    # all_dets2 = all_dets[1]
    # print(len(all_dets1), len(all_dets2))
    num = 0
    for i, dets in enumerate(all_dets1):
        # print(i)
        if i == 646:
            break
        image_id = i
        gallery_dicts1[image_id] = dict()
        # print(i)

        gallery_dicts1[image_id]['bbox'] = dets[0][:, :4]
        gallery_dicts1[image_id]['scores'] = dets[0][:, 4]
        gallery_dicts1[image_id]['feats'] = dets[0][:, 5:]
        gallery_dicts1[image_id]['pred_bbox'] = dets[0][:, :4]

        r, = dets[0][:, 4].shape
        list_num = []
        # img = Image.open('/home/cvlab3/Downloads/WRCAN-PyTorch/src/images{}_cam/'.format(x1) + id_to_img[i])
        img = cv2.imread('/home/cvlab3/Downloads/WRCAN-PyTorch/src/images{}_cam/'.format(x1) + id_to_img[i])
        for k in range(r):
            if gallery_dicts1[image_id]['scores'][k] > 0.2:
                # print('bbox',gallery_dicts1[image_id]['bbox'][i])

                numbers = gallery_dicts1[image_id]['bbox'][k]
                # print(numbers)
                list_num.append(numbers)

                l = int(numbers[0])
                t = int(numbers[1])
                r = int(numbers[2])
                b = int(numbers[3])
               # print((l,r), (r,b))
                cv2.rectangle(img, (l,t),(r,b),(0,0,255),3)
                # print('bbox', (numbers))
                # cropped_img = img.crop((numbers))
                # cropped_img.save("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_modify{}_result/{}_{}.jpeg".format(x1,i,num))
                num = num +1
        gallery_dicts1[image_id]['pred_bbox'] = np.array(list_num, dtype=object)

        # img.save("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images{}_result/{}_{}.jpeg".format(x1, i, num))
        cv2.imwrite("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_modify{}_result/{}_{}.jpeg".format(x1, i, num), img)  # save img

        # if gallery_dicts1[image_id]['scores']
        # cv2.imwrite("/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_result/{}.jpeg".format(i),
        #         gallery_dicts1[image_id]['bbox'])


    thres = 0.2


    ap = 0
    precision = {}
    recall = {}

    for image_id in range(img_num-1):

        query_box = gallery_dicts1[image_id]['pred_bbox']  # predicted bb
        query_box = [list(map(int, q)) for q in query_box]


        box = [b['bbox'] for b in test_det['annotations'] if b['image_id']==image_id]
        # boxs.append(box)


        query_gt_box =  box

        tp_num = get_max_iou(query_box, query_gt_box)

        # print('tp_num ', tp_num)
        if (len(query_gt_box) ==0) or (len(query_box)==0):
            # precision[image_id] = 0
            # recall = 0
            continue
        precision[image_id]=tp_num/len(query_box)
        recall[image_id]= tp_num/len(query_gt_box)
        # ap += precision*recall
    ap_num = 0

    for i in range(img_num-1):
        # if i+1:
        # print(precision[i+1],recall[i+1],recall[i])
        # if (precision[i+1]) and (recall[i+1]) and (recall[i]):
        try:
            ap += precision[i+1]*(recall[i+1]-recall[i])
            ap_num += 1
        except:
            pass


    map = ap/ap_num
    print('map',map)
        # exit()



