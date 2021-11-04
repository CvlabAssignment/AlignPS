# "annotations":
import collections
import json
from PIL import Image
import xml.etree.ElementTree as ET
import re
import glob

total_dict = {}
#fp =  open('test_new1.json', 'w')
#fp =  open('test_new1.json', 'w')
#fp2 = open('gallery.json', 'w')

fp = open('gallery_cam1.json', 'w')
fp2 = open('query_cam1.json' , 'w')


total_dict["categories"] = [{"id": 1, "name": "person", "supercategory": "object"}]
images_dict_list = []

annotations_dict_list = []
images_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_new/'
# images_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_x1_cam/'
images_file_names = glob.glob(images_path+'*.jpeg')
file_path = '/data1/KIST/data/20200904_KIST_DATA/'

num_index = 0
# l = []
# for idx, name in enumerate(images_file_names):
#     name = name.split('/')
#     cam_name, _= name[-1].split('|')
#     cam_name  = cam_name[3:5]
#     l.append(cam_name)
#
#     # eixt()
# print(collections.Counter(l))


for file_index, f in enumerate(images_file_names):
    f = f.split("/")
    images_dict = {"width": 1920, "height": 1080}
    images_dict["file_name"] = str(f[-1])
    # print(f[-1])
    images_dict["id"] = file_index
    print('i', images_dict)
    images_dict_list.append(images_dict)
    # print(f)
    f = f[-1].split("|")
    frame_no = int(f[1][:-5])
    file_name = f[0]
    # print(frame_no, file_name)
    tree = ET.parse(file_path + '카메라{}_FrameInfo.xml'.format(file_name[3:]))
    root = tree.getroot()
    child = root[0]
    # print(child[frame_no - 1].attrib, frame_no, file_name)
    for index, i in enumerate(child[frame_no - 1]):
        if i.tag.startswith('H'):
            numbers = re.findall(r'\d+', str(i.attrib))
            # print('n',numbers)
            img = Image.open(images_path + file_name + "|" + str(frame_no) + ".jpeg")
            annotations_dict = {"category_id": 1,"iscrowd": 0, "segmentation": []}
            annotations_dict["bbox"] = [numbers[1],numbers[3],numbers[5],numbers[7]]
            # print(file_name[3:5])
            annotations_dict["area"] = int(file_name[3:5])
            annotations_dict["image_id"] = file_index
            # print(num_index, file_index)
            annotations_dict["id"] = num_index
            num_index = num_index + 1
            print('a',annotations_dict)
            annotations_dict_list.append(annotations_dict)
            # print(i.attrib, numbers)
            # l = int(numbers[1])
            # t = int(numbers[3])
            # r = int(numbers[5])
            # b = int(numbers[7])
            # print(img_idx, area)
            # cropped_img = img.crop((l, t, r, b))
            # cropped_img.save('./images_x1_crop/' + str(file_name) + "_" + str(frame_no) + "_" + str(i.tag) + '.jpg')


total_dict["images"] = images_dict_list
total_dict["annotations"] = annotations_dict_list

#json.dump(total_dict, fp)