"""
Author: md Shamim
Description: this file collect each subject's pose information using openpose library
"""

# python packages
import os
import numpy as np

# project modules
from .. import config


# running openpose for casiaA dataset
def run_openpose(cheat_list):

    # considering each subject
    for cheat_ID in cheat_list:

        cheat_dir = os.path.join(config.data_3dcd_path(), cheat_ID)
        cheat_vid_list = os.listdir(cheat_dir)

        num_seq =  len(cheat_vid_list)
        #print("\n\n%s subject have: %d gait sequence vidoes" % (subject_id, num_seq))

        # considering each gait sequence
        for cheat_vid in cheat_vid_list:
            cheat_vid_dir = os.path.join(cheat_dir, cheat_vid)
            
            # save_dir for saving pose keypoints data
            save_dir = os.path.join(config.pose_3dcd_path(), cheat_ID, cheat_vid)
            os.makedirs(save_dir, exist_ok = True)

            # setting openpose directory
            os.chdir(config.openpose_path())

            print("\ncalculationg pose...")
            os.system("./build/examples/openpose/openpose.bin --image_dir " +  
                        cheat_vid_dir + " --number_people_max 1 " + " --write_json " +  
                        save_dir + " --display 0 --render_pose 0")



# getting pose data for all available subject using openpose library
def get_pose_data():
    
    # calculating total number of person have gait videos
    cheat_list = os.listdir(config.data_3dcd_path())
    print(cheat_list)
    run_openpose(cheat_list)


# run here
if __name__ == "__main__":
    get_pose_data()