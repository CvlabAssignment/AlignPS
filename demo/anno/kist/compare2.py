# "annotations":
import json
from PIL import Image
import xml.etree.ElementTree as ET
import re
import glob

# total_dict = {}
# fp =  open('test_new1.json', 'w')
# total_dict["categories"] = [{"id": 1, "name": "person", "supercategory": "object"}]
# images_dict_list = []
#
# annotations_dict_list = []
images_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_cam/'
images_path_x1 = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_x1_cam/'
images_file_names = glob.glob(images_path+'*.jpeg')
images_file_names_x1 = glob.glob(images_path+'*.jpeg')
file_path = '/data1/KIST/data/20200904_KIST_DATA/'
sort_list1 = []
sort_list2 = []
for l in images_file_names:
    b = l.split('|')
    f = l.split("_")
    # print('f', f)
    a = float(f[1][-2:] + '' + b[-1][:-4])
    # print(f[1][-2:])
    # a = int()
    sort_list1.append(int(a))
num_list1 = sorted(sort_list1)

print('set',set(images_file_names_x1)-set(images_file_names))

for l in images_file_names_x1:
    b = l.split('|')
    f = l.split("_")
    a = float(f[1][-2:] + '' + b[-1][:-4])
    sort_list2.append(a)
num_list2 = sorted(sort_list2)

# print(num_list1)
# num_list = sorted([int(l[3:4]+''+l[-8:-4]) for l in images_file_names])

# num_index = 0
for file_index, f in enumerate(num_list1):
    if num_list2[file_index] == f:
        pass
    else:
        print(file_index, f)
    # f = f.split("/")
    # images_dict = {"width": 1920, "height": 1080}
    # images_dict["file_name"] = str(f[-1])
    # # print(f[-1])
    # images_dict["id"] = file_index
    # print('i', images_dict)
    # images_dict_list.append(images_dict)
    # # print(f)
    # f = f[-1].split("|")
    # frame_no = int(f[1][:-5])
    # file_name = f[0]
    # print(frame_no, file_name)
    # tree = ET.parse(file_path + '카메라{}_FrameInfo.xml'.format(file_name[3:]))
    # root = tree.getroot()
    # child = root[0]
    # # print(child[frame_no - 1].attrib, frame_no, file_name)
    # for index, i in enumerate(child[frame_no - 1]):
    #     if i.tag.startswith('H'):
    #         numbers = re.findall(r'\d+', str(i.attrib))
    #         # print('n',numbers)
    #         img = Image.open(images_path + file_name + "|" + str(frame_no) + ".jpeg")
    #         annotations_dict = {"category_id": 1,"iscrowd": 0, "segmentation": []}
    #         annotations_dict["bbox"] = [numbers[1],numbers[3],numbers[5],numbers[7]]
            # print(file_name[3:5])


# json.dump(total_dict, fp)