"""
Author: Md Shamim
Description: generating frames from given video files
"""

# python packages
import cv2
import os

from .. import config

"""
# for training CASIA-B Dataset
def get_all_video_files_for_train ():
    ls_video_files = []
    grp = ["Nocheat", "LookLeft","LookRight" , "PocketSheet", "PantsSheet", "ExchPaper", "FaceCodes", "HandCodes"]

    for v_a in grp:
        ls_video_files += os.listdir(os.path.join(input_dir,v_a)) 
        #ls_video_files = ls_video_files[0].split(".")[0][-3:]

        

    return ls_video_files
print(get_all_video_files_for_train())
"""


# for training CASIA-B Dataset
def get_all_video_files_for_train ():
    group_list = ["grup" + str(i) for i in range(1,9)]
    ls_video_files = []

    for group_id in group_list:
        group_dir = os.path.join(config.dataset_3dcd_path(), group_id)
        cheat_list = os.listdir(group_dir)

        for cheat_type in cheat_list:
            cheat_dir = os.path.join(group_dir, cheat_type)
            
            cheat_vid_list = os.listdir(cheat_dir)
            for cheat_vid in cheat_vid_list:
                input_dir = os.path.join(cheat_dir, cheat_vid)
                ls_video_files.append(input_dir)

    return ls_video_files



# making video frames
def gen_video_frames(ls_video_files):
    for video_file in ls_video_files:
        #print(video_file)

        v = video_file.split("/")[-2:]

        out_cheat_dir = os.path.join(config.data_3dcd_path(), v[0])
        os.makedirs(out_cheat_dir, exist_ok = True)

        file_name = "_".join((v[1].split(".")[0]).split("_")[-2:])
        #print(file_name)

        out_vid_dir = os.path.join(out_cheat_dir, file_name)
        os.makedirs(out_vid_dir, exist_ok = True)

        # making all frame sequence per subject
        # capturing video
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()

        if(vidcap.isOpened() == True):
            count = 0
            success = True

        while success:
            success, image = vidcap.read()
            count += 1
            print('reading a new frame: ', success)

            if(success == True):
                #save frame as JPEG file
                cv2.imwrite(os.path.join(out_vid_dir, ("%03d.jpg") % count), image)  


# getting associated video files
ls_video_files = get_all_video_files_for_train ()

# making video frames for given videos
gen_video_frames(ls_video_files)