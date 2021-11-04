import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
# src_file_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images/'
# dst_file_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_cam/'

src_file_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/images_new_x1/'
dst_file_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/src/im/'

file_names = os.listdir(src_file_path)
# num_list = sorted([int(l[:-4]) for l in file_names])
# num_list_reverse = reversed(num_list)
# print(num_list)
# print(file_names)
#
for fn in file_names:
    # print(fn)
    src = src_file_path + fn
    dst = dst_file_path + 'cam{}'.format(fn[3:])

    #     # "{:010d}{}".format(i, '.png')
    # # print(dst)
    # # dst = str(i) + '.jpg'
    # # dst = os.path.join
    print('{} ==> {}'.format(src, dst))
    os.rename(src, dst)