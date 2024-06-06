import json
import os
import numpy as np

#GES = ["up_tap","down_up","left_down","left_right"]

GES = {"left_down_up","tap_tap_right","down_left","down_left_up_right","down_right_tap_left_up","right_up_left_tap","up_right_down_tap"}

ges = 0
bou = 0
non_ges = 0


for gesture in GES:
    File_root = f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/continuous_eval2/{gesture}/"
    non_ges_j = 0
    cut = gesture.split("_")
    first_ges = cut[0]
    second_ges = cut[1]
    print("1st_ges: ",first_ges)
    print("2nd_ges: ",second_ges)
    f_ges_num = 0
    s_ges_num = 0
    l = len(os.listdir(File_root))




    for activitiy in range(l):
        sample_dir_path = File_root + str(activitiy + 1) + "/"
        acty_seq_len = len(os.listdir(sample_dir_path))
        frameNum = 0
        swich = 0
        non_ges += 1
        non_ges_j = 0
        f_flag = 1
        sec_flag = 0

        #fir_ges_len = os.listdir(f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges/{first_ges}/")
        #sec_ges_len = os.listdir(f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges/{second_ges}/")

        if first_ges == "tap":
            fir_ges_len = len(os.listdir(
                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{first_ges}/"))
        else:
            fir_ges_len = len(os.listdir(
                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{first_ges}/"))

        if second_ges == "tap":
            sec_ges_len = len(os.listdir(
                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{second_ges}/"))
        else:
            sec_ges_len = len(os.listdir(
                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{second_ges}/"))


        for json_num in range(0, acty_seq_len):
            print("start json No.", json_num + 1)
            json_file_path = sample_dir_path + '/' + str(json_num + 1) + '.json'

            with open(json_file_path, 'r') as load_path:
                load_dict = json.load(load_path)
                frameNum += 1
                print("framenum", frameNum)
                label = load_dict['label']
                pointcloud = load_dict["PointCloud"]
                d = {
                    "PointCloud": pointcloud,
                    "label": label
                }

                if (swich != label):
                    if (label == 1):
                        ges += 1
                        ges_j = 0

                    elif (label == 2):
                        bou += 1
                        bou_j = 0
                        if swich == 1:
                            f_flag=0
                            sec_flag = 1


                    elif (label == 0):
                        non_ges += 1
                        non_ges_j = 0
                        if swich == 2:
                            if sec_flag == 1:
                                sec_flag = 0


                # print(label)
                if (label == 0):
                    """
                    os.makedirs(
                        f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/cut_conti/non_gesture/{non_ges}",
                        exist_ok=True)
                    with open(
                            f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/cut_conti/non_gesture/{non_ges}/{non_ges_j + 1}.json",
                            "w") as f:
                        json.dump(d, f)
                    """
                    non_ges_j += 1
                    swich = 0



                elif (label == 1):
                    if(f_flag == 1):

                        if first_ges == "tap":
                            os.makedirs(
                                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{first_ges}/{str(fir_ges_len + 1)}/",exist_ok=True)
                            with open(
                                    f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{first_ges}/{str(fir_ges_len + 1)}/{ges_j + 1}.json",
                                    "w") as f:
                                json.dump(d, f)
                            swich = 1
                            ges_j += 1
                        else:
                            os.makedirs(
                                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{first_ges}/{str(fir_ges_len + 1)}/",exist_ok=True)
                            with open(
                                    f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{first_ges}/{str(fir_ges_len + 1)}/{ges_j + 1}.json",
                                    "w") as f:
                                json.dump(d, f)
                            swich = 1
                            ges_j += 1

                    if(sec_flag == 1):
                        if second_ges == "tap":
                            os.makedirs(
                                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{second_ges}/{str(sec_ges_len + 1)}/",exist_ok=True)
                            with open(
                                    f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/{second_ges}/{str(sec_ges_len + 1)}/{ges_j + 1}.json",
                                    "w") as f:
                                json.dump(d, f)
                            swich = 1
                            ges_j += 1
                        else:
                            os.makedirs(
                                f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{second_ges}/{str(sec_ges_len + 1)}/",exist_ok=True)
                            with open(
                                    f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges3_cut_conti/swipe_{second_ges}/{str(sec_ges_len + 1)}/{ges_j + 1}.json",
                                    "w") as f:
                                json.dump(d, f)
                            swich = 1
                            ges_j += 1





                elif (label == 2):
                    """
                    os.makedirs(
                        f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/cut_conti/bound/{bou}",
                        exist_ok=True)
                    with open(
                            f"C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/cut_conti/bound/{bou}/{bou_j + 1}.json",
                            "w") as f:
                        json.dump(d, f)
                    """
                    swich = 2
                    bou_j += 1

