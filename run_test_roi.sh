# CUHK-SUSU ROI-AlignPS
TESTPATH='faster_rcnn_r50_caffe_c4_1x_cuhk_single_two_stage17_6_nae1'
TESTNAME='cuhk_roi_alignps.pth'

# Make sure the model path is work_dirs/TESTPATH/TESTNAME, tesults_1000.pkl will be saved in work_dirs/TESTPATH
#epoch_${i}
./tools/dist_test_d.sh ./configs/person_search/${TESTPATH}.py work_dirs/${TESTPATH}/${TESTNAME} 1 --out work_dirs/${TESTPATH}/results_1000.pkl
#./tools/dist_test_d.sh /home/cvlab3/Downloads/AlignPS/configs/person_search/${TESTPATH}.py work_dirs/${TESTPATH}/${TESTNAME} 1 --out /home/cvlab3/Downloads/AlignPS/work_dirs/${TESTPATH}/results_1000.pkl
echo '------------------------'
python ./tools/test_results_psd.py ${TESTPATH}
echo $TESTPATH
