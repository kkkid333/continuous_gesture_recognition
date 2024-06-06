
import time
start_time=time.process_time()
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
import random
seed=42


import torch
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

seed = 42
eps = 0.1
min_samples = 11
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch version --",torch.__version__)
print("Using device ", device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
    print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
else:
    print("No GPU available.")

def load_traink_fold_data(self):
    import numpy as np
    import torch

    # Load the dataset
    dataset = np.load("training_data.npy", allow_pickle=True)

    # Convert from numpy to torch
    print("dataset",dataset.shape)
    print(dataset)
    X = torch.Tensor([i[0] for i in dataset])
    y = torch.Tensor([i[1] for i in dataset])

    # Set the number of folds
    k = 5

    # Calculate the fold size
    fold_size = len(X) // k

    # Lists to store the training and testing sets
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # Perform k-fold cross-validation
    for i in range(k):
        # Calculate the starting and ending indices for the current fold
        start = i * fold_size
        end = (i + 1) * fold_size

        # Split the data into training and testing sets for the current fold
        X_test = X[start:end]
        y_test = y[start:end]

        if i == 0:
            X_train = X[end:]
            y_train = y[end:]
        elif i == k - 1:
            X_train = X[:start]
            y_train = y[:start]
        else:
            X_train = torch.cat((X[:start], X[end:]), dim=0)
            y_train = torch.cat((y[:start], y[end:]), dim=0)

        # Add the sets to the lists
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    # Print the shapes of the resulting sets for each fold
    for i in range(k):
        print("Fold", i + 1)
        print("X_train shape:", X_train_list[i].shape)
        print("X_test shape:", X_test_list[i].shape)
        print("y_train shape:", y_train_list[i].shape)
        print("y_test shape:", y_test_list[i].shape)

def denoising_DBSCAN(x_para, y_para, z_para, doppler_para, snr_para,eps,min_samples):
    x_pos = np.array(x_para)
    y_pos = np.array(y_para)
    z_pos = np.array(z_para)
    doppler_pos = np.array(doppler_para)
    snr_pos = np.array(snr_para)

    # Define the DBSCAN parameters
    eps = eps  # Maximum distance between points to be considered neighbors
    min_samples = min_samples# Minimum number of points required to form a cluster
    #print("gesture_name,eps_t,min_samples_t",eps_t,min_samples_t)

    # Perform DBSCAN on the input data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(np.column_stack((x_pos, y_pos, z_pos)))

    # Identify the good points and their corresponding doppler and snr values
    good_points_mask = labels != -1  # -1 indicates a noisy point
    good_x_pos = x_pos[good_points_mask]
    good_y_pos = y_pos[good_points_mask]
    good_z_pos = z_pos[good_points_mask]
    good_doppler_pos = doppler_pos[good_points_mask]
    good_snr_pos = snr_pos[good_points_mask]
    return good_x_pos, good_y_pos, good_z_pos, good_doppler_pos, good_snr_pos

#added by Eric to calculate Average SNR
def chech_avr_gest_snr(sample_dir_path):

    gesture_mean_box = []

    acty_seq_len = len(os.listdir(sample_dir_path))
    for json_num in range(0, acty_seq_len):

        # json_file_path = sample_dir_path + '/' + str(json_num + 1) + '.json'
        json_file_path = sample_dir_path + '/' + str(json_num + 1) + '.json'

        with open(json_file_path, 'r') as load_path:
            load_dict = json.load(load_path)
            PointCloud = load_dict['PointCloud']

            pcBufPing = np.array(PointCloud)
            x_pos = pcBufPing[0]
            snr_pos = pcBufPing[4]
            if len(x_pos) == 1150:

                continue
            else:
                frame_snr = np.array(snr_pos)
                avg_frame_snr = np.mean(frame_snr)
                # print("avg_frame_snr", avg_frame_snr)
                gesture_mean_box.append(avg_frame_snr)
    gesture_mean_box = np.array(gesture_mean_box)
    #print("len(gesture_mean_box", len(gesture_mean_box), gesture_mean_box, sample_dir_path)

    avr_gest_threshold = np.mean(gesture_mean_box)
    #print("first avg gesture",avr_gest_threshold)
    return avr_gest_threshold

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

framelength = 5
pointlength = 1150


import os
import json
import numpy as np

class Gestures():

    #STOP= "stop"
    #WALK= "walk"




    """
    SWIPE_DOWN = "swipe_down"
    SWIPE_LEFT= "swipe_left"
    SWIPE_RIGHT  = "swipe_right"
    SWIPE_UP = "swipe_up"
    TAP = "tap"


    LABELS = {
        SWIPE_DOWN : 0, SWIPE_LEFT:1, SWIPE_RIGHT:2, SWIPE_UP:3, TAP:4
    }
    """


    DOWN_UP = "down_up"
    LEFT_DOWN = "left_down"
    LEFT_RIGHT = "left_right"
    UP_TAP = "up_tap"
    NON_GESTURE = "non_gesture"
    STOP = "stop"

    LABELS = {
        DOWN_UP:0, LEFT_DOWN:1, LEFT_RIGHT:2, UP_TAP:3,STOP:4
    }



    global class_number
    class_number = 3
    training_data = []



    def read_database(self, gesture_name):
        dataset = []
        #gest_sub_dir = "C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/sign/"
        #gest_sub_dir = "C:/Users/Eric/Desktop/FOLDER41_42/"
        #gest_sub_dir = "C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/labeled_data/"
        gest_sub_dir = "C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/origin_data/origin_data/"
        #non_ges_path = "C:/Users/PC_User/PycharmProjects/mmwave/mmwave/learn_data/cls50/non_gesture/"
        gesture_path = gest_sub_dir + gesture_name

        l0 = 0
        l1 = 0
        l2 = 0
        total =0

        if gesture_name == "stop":
            l = 101
        else:
            l = 80





        for gest_num in range(0,l):#gestureの1とか2とか
            each_gesture_path = gesture_path + "/" + str(gest_num+1)+"/"
            #print(gest_num)
            if not os.path.exists(each_gesture_path):
                continue

            activity_seq_len = len(os.listdir(each_gesture_path))#フレームの長さ（今見ているジェスチャーの

            FrameNumber = 0
            gestureNum = 0
            #framelength2 = 30
            pointlength = 1150
            maxopoints = 1150
            #gesturedata = np.zeros((framelength, 4, pointlength))
            one_shot_data = []

            for json_num in range(0, activity_seq_len):
                each_frame_path = each_gesture_path + '/' + str(json_num + 1) + '.json'
                flag = 0

                if activity_seq_len < framelength:
                    leave = framelength - activity_seq_len
                    flag = 1

                if not os.path.exists(each_frame_path):
                    continue

                with open(each_frame_path, 'r') as load_path:
                    load_dict = json.load(load_path)
                    PointCloud = load_dict['PointCloud']
                    if gesture_name == "stop":
                        label = 0
                    else:
                        label = load_dict["label"]

                    #print(label)
                    pcBufPing = np.array(PointCloud)


                    #ここからfor分までの作業は足りない座標を０で埋めている
                    x = np.zeros(pointlength)
                    y = np.zeros(pointlength)
                    z = np.zeros(pointlength)
                    doppler = np.zeros(pointlength)
                    #snr = np.zeros(300)

                    x_pos = pcBufPing[0]
                    y_pos = pcBufPing[1]
                    z_pos = pcBufPing[2]
                    doppler_pos = pcBufPing[3]
                    #snr_pos = pcBufPing[4]

                    if len(x_pos) >maxopoints:#点群のmax
                        continue

                    for i in range(0, int(len(x_pos))):
                        if (x_pos[i] <= 2) and (x_pos[i] >= -2):
                            if (z_pos[i] <= 1.5) and (z_pos[i] >= -1):
                                if (y_pos[i] <= 2.0):
                                    x[i] = x_pos[i]
                                    y[i] = y_pos[i]
                                    z[i] = z_pos[i]
                                    doppler[i] = doppler_pos[i]

                        #snr[i] = snr_pos[i]
                    #ここまで




                    framedata = [x, y, z,doppler,label]#１フレームの情報
                    #print("framedata",famedata)
                    #print("framedata_shape",framedata.shape)
                    #print("framedata_shape",np.array(framedata[0]).shape)
                    if(FrameNumber >= framelength):
                        one_shot_data.pop(0)
                        #print("oen_shot:" , np.array(one_shot_data).shape)

                    one_shot_data.append(framedata)#フレームとラベルの組これはシャッフルしてはいけない
                    if flag:
                        if json_num == activity_seq_len -1:
                            for i in range(0,leave):
                                one_shot_data.append([np.zeros(pointlength), np.zeros(pointlength), np.zeros(pointlength),np.zeros(pointlength),0])

                            l0 += framelength
                            dataset.append(one_shot_data.copy())
                            total += 1

                    if(FrameNumber >= framelength-1):
                        #print("one_shot", [row[4] for row in one_shot_data])
                        dataset.append(one_shot_data.copy())
                        #print("dataset: ", np.array(dataset).shape)
                        #print("dataset_len: ",len(dataset))
                        #print("dataset_0",[row[4] for row in dataset[-1]])
                        total += 1
                        for i in one_shot_data:
                            la = i[4]
                            if la == 0:
                                l0 += 1
                            if la == 1:
                                l1 += 1
                            if la == 2:
                                l2 += 1
                        #print("one_shot",[row[4] for row in one_shot_data])

                    FrameNumber += 1

                    #print("one_shot_data: ", np.array(one_shot_data).shape)



            #dataset.append(one_shot_data)

        #print("dataset: ", dataset)

        a0 = 0
        a1 = 0
        a2 = 0

        #"""

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("dataset: ", np.array(dataset).shape)

        for d in dataset:
            #print("dataset",[row[4] for row in d])
            #print("dataset_elem; ", [row[4] for row in d])
            for i in range(len(d)):
                la = d[i][4]
                if la == 0:
                    a0 += 1
                if la == 1:
                    a1 += 1
                if la == 2:
                    a2 += 1
        print("a0: ", a0)
        print("a1: ", a1)
        print("a2: ", a2)

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #"""

        print("label0", l0)
        print("label1", l1)
        print("label2", l2)

        #print("label_total: ", l0+l1+l2)
        number_of_samples = len(dataset)
        print("dataset_count: ", number_of_samples, total)
        #dataset.shapedata
        #print(dataset.shape)

        return dataset, number_of_samples

    def load_data(self):
        total = 0
        for label in self.LABELS:
            trainset, number_of_samples = self.read_database(label)

            for data in trainset: #ここでラベリングをしている
                print("data.shape: ", np.array(data).shape)
                print("label.shape", np.array(label).shape)
                print("label", self.LABELS[label])

                self.training_data.append(
                    [np.array(data), np.array([self.LABELS[label]])]) #save data and assign label
                #print(self.training_data)
                #print(np.array(self.training_data))

            total = total + number_of_samples
            #print("training_data ", np.array(self.training_data).shape)
            print(label, number_of_samples)


        print("Total number:", total)

        #np.random.shuffle(self.training_data)
        #np.save("training_data.npy", self.training_data)



    def load_data2(self):#ここでサンプリング方法をかえる
        total = 0
        all_data = []
        for label in self.LABELS:
            print(label)
            trainset, number_of_samples = self.read_database(label)
            #print("trainset: ", np.array(trainset).shape, len(trainset))

            #all_data += trainset

            #print("----------------------------------------------")
            for data in trainset:
                #print("data_shape: ",np.array(data).shape)
                #print()
                #print("lanels: ", [l[4] for l in data])
                #print("x: ", [l[0][0] for l in data])
                all_data.append(data) #save data and assign label
            #print("-----------------------------------------------")


            total = total + number_of_samples
            print(label, number_of_samples)

        print("Total number:", total)
        #np.random.shuffle(self.training_data)
        #np.save("training_data.npy", self.training_data)
        return all_data





class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.rnn = nn.RNN(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out


class RNN_GRU(nn.Module):
    def __init__(self):
        super(RNN_GRU, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.gru = nn.GRU(300 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        # self.gru = nn.GRU(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out


class RNN_LSTM(nn.Module):
    def __init__(self):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1011* frame_parameters, neurons_num, self.num_layers, batch_first=True)
        # self.lstm = nn.LSTM(1150 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num *60, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print("h0, h0.shape,",h0,h0.shape)
        # print("c0, c0.shape,",c0, c0.shape)
        # print("x_passed_to_LSTM, X_passed_.shape,", x, x.shape)
        # print("x.size(0)",x.size(0))
        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)

        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out






class LSTM_LIN(nn.Module):
    def __init__(self):
        super(LSTM_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.lstm = nn.LSTM(frame_parameters*pointlength, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num*framelength, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, framelength*class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)xはテンソル
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = out.view(-1,framelength,3)

        return out

    def name(self):
        return "LSTM_RIN"

class LSTM_LIN_2(nn.Module):
    def __init__(self):
        super(LSTM_LIN_2, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.lstm = nn.LSTM(frame_parameters*pointlength, neurons_num, self.num_layers, batch_first=True,dropout=1)
        self.fc1 = nn.Linear(neurons_num*framelength, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, framelength*class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)xはテンソル
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = out.view(-1,framelength,3)

        return out


    def name(self):
        return "LSTM_RIN_dropout"




class GRU_LIN(nn.Module):
    def __init__(self):
        super(GRU_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.gru = nn.GRU(1011 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        # self.gru = nn.GRU(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 60, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class RNN_LIN(nn.Module):
    def __init__(self):
        super(RNN_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        # self.rnn = nn.RNN(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.rnn = nn.RNN(300 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 120, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

def cos_sim(X,Y):
    print(type(X))
    X = np.array(X.cpu())
    Y = Y.cpu().numpy()
    return np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

def start_ml(layer, nodes_number, epochs, optimizer_step_value, VAL_PCT, parameters, break_limit):
    global neurons_num
    global num_layers
    global frame_parameters

    num_layers = layer
    neurons_num = nodes_number
    frame_parameters = parameters

    net = LSTM_LIN().to(device)
    print("model: ",net.name())
    gestures = Gestures()
    dataset= np.array(gestures.load_data2())
    #print()

    # Load data from .npy
    #dataset = np.load("training_data.npy", allow_pickle=True)

    print("dataset: ", dataset.shape,len(dataset))

    # Convert from numpy to torch

    y = []
    x = []
    new_framedata = []
    new_seq = []
    frame_label = []

    label1 = 0
    label2 = 0
    label0 = 0

    #ここがなにかおかしい
    for seq in dataset:
        #print("seq",len(seq))
        #print([l[4] for l in seq])
        for framedata in seq:
            #print("framedata",len(framedata))
            #print(framedata[0][4])
            #print("framedata: ", np.array(framedata).shape)
            la= framedata[4]
            #print(la)
            new_framedata.append([framedata[0],framedata[1],framedata[2],framedata[3]])
            #print("framedata[0][4]: ",framedata[0][4])
            frame_label.append(framedata[4])
            if la == 0:
                label0 += 1
            if la == 1:
                label1 += 1
            if la == 2:
                label2 += 1
            #print("frame_label",framedata[0][4])
            #print('new',np.array(new_framedata).shape)
            #new_seq.append(new_framedata)
            #print("end_seq")
        x.append(new_framedata)
        y.append(frame_label)
        #print(y)
        #("frame_label__shape: ", np.array(frame_label).shape)
        new_framedata = []
        frame_label = []
        #print("x",len(x))
        #print("y",len(y))
        #print(y)
    #print("x", np.array(x).shape)
    #print("y", np.array(y).shape)

    print("gesture_label_count: ", label1)
    print("non_gesture_label_count: ", label0)
    print("boundary_label_count: ", label2)

    w1 = 1.0
    w2 = round(label1/label2,3)
    w0 = round(label1/label0,3)


    weights = torch.tensor([w0,w1,w2])
    weights = weights.to(device)

    weights2 = torch.tensor([1/label0, 1/label1, 1/label2])
    weights2 = weights2.to(device)



    t = label0 + label2 + label1

    weights3 = torch.tensor([(t / (t - label0)), (t / (t - label1)), (t / (t - label2))])
    weights3 = weights3.to(device)

    print("w0", t / (t - label0))
    print("w1", t / (t - label1))
    print("w2", t / (t - label2))

    print("label_total",t)

    #exit()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=VAL_PCT, random_state=100,shuffle=True)

    train_X = torch.tensor(X_train,dtype=torch.float32)
    train_y = torch.tensor(y_train,dtype=torch.float32)

    test_X = torch.tensor(X_test,dtype=torch.float32)
    test_y = torch.tensor(y_test,dtype=torch.float32)

    """
    X = torch.Tensor(x)
    y = torch.Tensor(y)

    # Divide dataset to training set and test set
    val_size = int(len(dataset) * VAL_PCT)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]
    """

    print("test_X_shape: ", np.array(test_X).shape)
    print("test_y_shape: ", np.array(test_y).shape)

    print("Trainingset:", len(train_X))
    print("Testset", len(test_X))

    criterion = nn.CrossEntropyLoss(weight=weights2)
    optimizer = optim.Adam(net.parameters(), lr=optimizer_step_value)

    loss_list = []
    accuracy_list = []
    iteration_list = []
    tot_accuracy = 0
    count = 0
    earlystopping = EarlyStopping(patience=3, verbose=True)

    print("Layers: ", layer, "Neurons: ", neurons_num)
    print("----------start train------------")
    for epoch in range(epochs):  # 3 full passes over the data


        for i in range(len(train_X)): #labelを考える
            #print("len_train_x: ",len(train_X))
            data = train_X[i].to(device=device)
            targets = train_y[i].to(device=device)
            targets = torch.tensor(targets, dtype=torch.long)
            #targets = targets.view(15,1)
            #print("data shape,before passing to view and LSTM", data.shape)
            #print("target shape before passing to view and LSTM: ",targets.shape)
            #print("data shape: ", data)
            #print("target: ", targets)

            scores = net(data.view(-1, framelength, pointlength*frame_parameters))#ここでネットワークにぶち込んでる
            #print("the score /score shape:", scores, scores.shape)
            scores = scores.view(framelength,3)
            #print("score shape: ", scores.shape)
            #print("target: ", targets)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()
            count += 1

            tot_number = 0
            tot_correct = 0
            cos_sum = 0
            # Every 10 iterations calculate accuracy
            #print("count % 200", count % 200)
            if count %10000 == 0:
                with torch.no_grad():
                    # Calculate accuracy for each gesture and total accuracy 精度を計算している
                    """
                    for label in Gestures.LABELS:#ジェスチャーごとに繰り返している
                        correct = 0
                        number = 0
                        for a in range(len(test_X)):
                            print("len(test_x): ", len(test_X))
                            print("test_x: ", test_X.shape)
                            if test_y[a] == Gestures.LABELS[label]:
                                X = test_X[a].to(device=device)
                                y = test_y[a].to(device=device)

                                output = net(X.view(-1, framelength, pointlength*frame_parameters))
                                print("output is",output)
                                for idx, i in enumerate(output):
                                    #print("output,idx, i, torch.argmax(i) ", output, idx, i, torch.argmax(i))
                                    if torch.argmax(i) == y[idx]:
                                        tot_correct += 1
                                        correct += 1
                                    number += 1
                                    tot_number += 1

                    tot_accuracy = round(tot_correct / tot_number, 3)
                    """

                    tot_classified2 = np.zeros([class_number, class_number])

                    la0 = 0
                    la1 = 0
                    la2 = 0

                    for a in range(len(test_X)):
                        #print("len(test_x): ", len(test_X))
                        #print("test_x: ", test_X.shape)
                        seq_x = test_X[a]
                        seq_y = test_y[a]
                        #print("seqX_shape: ", np.array(seq_x).shape)#(15,4,50)
                        #print("seqy_shape: ", np.array(seq_y).shape)#(15,)

                        X = seq_x.to(device=device)
                        y = seq_y.to(device=device)

                        output = net(X.view(-1, framelength, pointlength * frame_parameters))
                        #print("output is", output)
                        #print("output_shape: ", output.dim())#(15*3)

                        for idx, i in enumerate(output):# enumerateはリストのインデックスと値を返す iにはseqeunce(frameの集まりが入っている
                            #print("idx: ",idx)
                            #print("i: ", i) #(15*3)
                            #print("i_shape: ", i.dim()) #(15*3)
                            seq = []
                            for j in range(0,len(i)):
                                #print("argmax(j): ", torch.argmax(i[j]))
                                #print("y[idx]: ",y[idx])
                                #print(f"{j}番目")
                                if y[idx] == 0:
                                    la0 += 1
                                if y[idx] == 1:
                                    la1 += 1
                                if y[idx] == 2:
                                    la2 += 1
                                seq.append(torch.argmax(i[j]).to(torch.int))
                                # print("output,idx, i, torch.argmax(i) ", output, idx, i, torch.argmax(i))
                                if torch.argmax(i[j]) == y[idx]:
                                    tot_correct += 1
                                    # correct += 1

                                tot_number += 1
                                tru = y[idx].to(dtype=torch.int)
                                ref = torch.argmax(i[j]).to(dtype=torch.int)
                                #print("true_label = ",tru)
                                #print("ref_label = ",ref)
                                tot_classified2[tru][ref] += 1

                                #print("tot_numbertlive: ", tot_number)
                                #print("tot_correctlive: ", tot_correct)
                            #cos_sum += cos_sim(seq,y)

                    tot_accuracy = round(tot_correct / tot_number, 3)
                    #print("tot_accuracy: ", tot_accuracy)
                    print(tot_classified2)
                    print("label0: ", tot_classified2[0][0] + tot_classified2[0][1] + tot_classified2[0][2])
                    print("label1: ", tot_classified2[1][0] + tot_classified2[1][1] + tot_classified2[1][2])
                    print("label2: ", tot_classified2[2][0] + tot_classified2[2][1] + tot_classified2[2][2])

                    print("all accuracy: ", tot_accuracy)

                    non_ges_recall = tot_classified2[0][0]/(tot_classified2[0][0] + tot_classified2[0][1] + tot_classified2[0][2])
                    ges_recall = tot_classified2[1][1] / (
                                tot_classified2[1][0] + tot_classified2[1][1] + tot_classified2[1][2])
                    bou_recall = tot_classified2[2][2] / (
                                tot_classified2[2][0] + tot_classified2[2][1] + tot_classified2[2][2])

                    non_ges_pre = tot_classified2[0][0]/(tot_classified2[0][0] + tot_classified2[1][0] + tot_classified2[2][0])
                    ges_pre = tot_classified2[1][1] / (
                                tot_classified2[0][1] + tot_classified2[1][1] + tot_classified2[2][1])
                    bou_pre = tot_classified2[2][2] / (
                                tot_classified2[0][2] + tot_classified2[1][2] + tot_classified2[2][2])

                    print("non_gesture recall: ", non_ges_recall)
                    print("gesture recall: ", ges_recall)
                    print("boundary recall: ", bou_recall)

                    print("non_gesture precisin: ",non_ges_pre)
                    print("gesture precisin: ", ges_pre)
                    print("boundary precisin: ", bou_pre)

                    print("non_ges F1 score: ",2*non_ges_pre*non_ges_recall/(non_ges_pre+non_ges_recall))
                    print("ges F1 score: ", 2*ges_pre*ges_recall/(ges_pre+ges_recall))
                    print("boundary F1 score: ", 2*bou_pre*bou_recall/(bou_pre+bou_recall))

                    """
                    print("la0", la0)
                    print("la1", la1)
                    print("la2", la2)
                    """




                    #print("---------------tot_correct: ", tot_correct)
                    #print("---------------tot_num: ", tot_number)
                    #cos_accuracy = round(cos_sum / len(test_X), 3)
                    #exit()

                loss = loss.data
                loss = loss.cpu()
                loss_list.append(loss)
                iteration_list.append(count)
                accuracy_list.append(tot_accuracy)

        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss, tot_accuracy * 100))
        earlystopping(loss,net)
        if earlystopping.early_stop:  # ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

        if tot_accuracy * 100 > break_limit:
            break

    with torch.no_grad():
        net.eval()
        # Calculate accuracy for each gesture and total accuracy 最終的な精度
        tot_number = 0
        tot_correct = 0
        #tot_propability = np.zeros([class_number, class_number + 1])  # propability for every output
        tot_classified = np.zeros([class_number, class_number])  # Predicted labels matrix0

        for a in range(len(test_X)):
            # print("len(test_x): ", len(test_X))
            # print("test_x: ", test_X.shape)
            seq_x = test_X[a]
            seq_y = test_y[a]

            X = seq_x.to(device=device)
            y = seq_y.to(device=device)

            output = net(X.view(-1, framelength, pointlength * frame_parameters))
            # print("output is", output)
            #print("output_shape: ", output.dim())#(


            for idx, i in enumerate(output):  # enumerateはリストのインデックスと値を返す iにはseqeunce(frameの集まりが入っている
                # print("idx: ",idx)
                # print("i: ", i)
                seq = []
                label_reff = F.softmax(i, dim=1)
                #print("reffer: ",label_reff.dim(), label_reff)
                #label_reff = label_reff.cpu()
                #label_reff = label_reff.numpy()#15*3
                #print("label_reff: ", label_reff)#15*3
                #print("y: ", y)#(15,)

                for j in range(0, len(i)):
                    # print("argmax(j): ", torch.argmax(i[j]))
                    # print("y[idx]: ",y[idx])
                    #seq.append(torch.argmax(i[j]).to(torch.int))
                    # print("output,idx, i, torch.argmax(i) ", output, idx, i, torch.argmax(i))
                    #print(type(y[idx]))
                    #print(type(torch.argmax(label_reff[j])))

                    if torch.argmax(label_reff[j]) == y[idx]:
                        tot_correct += 1
                        # correct += 1
                    tot_number += 1
                    tru = y[idx].to(dtype=torch.int)
                    ref = torch.argmax(label_reff[j]).to(dtype=torch.int)

                    #print("tru",type(tru),tru)
                    #print("ref", type(ref), ref)

                    #print(type(tot_classified[tru][ref]))
                    tot_classified[tru][ref] += 1

                    # print("tot_numbertlive: ", tot_number)
                    # print("tot_correctlive: ", tot_correct)

                # cos_sum += cos_sim(seq,y)


        tot_accuracy = round(tot_correct / tot_number, 3)

        print("---------------tot_correct: ", tot_correct)
        print("---------------tot_num: ", tot_number)

        tot_accuracy = round(tot_correct / tot_number, 3) * 100
        print("Total accuracy: ", tot_accuracy)
        non_ges_recall = tot_classified[0][0] / (tot_classified[0][0] + tot_classified[0][1] + tot_classified[0][2])
        ges_recall = tot_classified[1][1] / (
                tot_classified[1][0] + tot_classified[1][1] + tot_classified[1][2])
        bou_recall = tot_classified[2][2] / (
                tot_classified[2][0] + tot_classified[2][1] + tot_classified[2][2])

        non_ges_pre = tot_classified[0][0] / (tot_classified[0][0] + tot_classified[1][0] + tot_classified[2][0])
        ges_pre = tot_classified[1][1] / (
                tot_classified[0][1] + tot_classified[1][1] + tot_classified[2][1])
        bou_pre = tot_classified[2][2] / (
                tot_classified[0][2] + tot_classified[1][2] + tot_classified[2][2])

        print("non_gesture recall: ", non_ges_recall)
        print("gesture recall: ", ges_recall)
        print("boundary recall: ", bou_recall)

        print("non_gesture precisin: ", non_ges_pre)
        print("gesture precisin: ", ges_pre)
        print("boundary precisin: ", bou_pre)

        print("non_ges F1 score: ", 2 * non_ges_pre * non_ges_recall / (non_ges_pre + non_ges_recall))
        print("ges F1 score: ", 2 * ges_pre * ges_recall / (ges_pre + ges_recall))
        print("boundary F1 score: ", 2 * bou_pre * bou_recall / (bou_pre + bou_recall))

        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:16.1f}'.format})

        # save results
        with open("results.txt", 'a') as f:
            print(tot_accuracy, file=f)

        print("行が正解ラベルで列が予測")
        print(tot_classified)
        print("label0: ", tot_classified[0][0] + tot_classified[0][1] + tot_classified[0][1])
        print("label1: ", tot_classified[1][0] + tot_classified[1][1] + tot_classified[1][2])
        print("label2: ", tot_classified[2][0] + tot_classified[2][1] + tot_classified[2][2])
        # print("save txt")
        #np.savetxt("propability matrix.txt", tot_propability, fmt='%f')
        #np.savetxt("confusion matrix.txt", tot_classified, fmt='%f')

        #torch.save(net.state_dict(), "point_1150_denoise_LSTM_LIN_win40.pth")

    # visualization loss
    plt.plot(iteration_list, loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("LSTM: Loss vs Number of iteration")
    plt.show()

    # visualization accuracy
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("LSTM: Accuracy vs Number of iteration")
    plt.show()


#layers_number = 2
nodes_array = [32]
epochs = 10  # number of epochs
optimizer_step_value = 0.0001  # optimizer step value for pytorch
test_percent = 0.1  # what percentage of the database will be used for the set test
parameters = 4
# parameters = 3
break_limit = 95
layer = 3

nodes_number =64
start_ml(layer, nodes_number, epochs, optimizer_step_value, test_percent, parameters, break_limit)

end_time=time.process_time()
execution_time=end_time-start_time
print("Execution time:",execution_time,"seconds")

