
import time
start_time=time.process_time()
import numpy as np
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


import os
import json
import numpy as np

class Gestures():

    #STOP= "stop"
    #WALK= "walk"



    SWIPE_DOWN = "swipe_down"
    SWIPE_LEFT= "swipe_left"
    SWIPE_RIGHT  = "swipe_right"
    SWIPE_UP = "swipe_up"
    TAP = "tap"


    LABELS = {
        SWIPE_DOWN : 0, SWIPE_LEFT:1, SWIPE_RIGHT:2, SWIPE_UP:3, TAP:4,
    }

        


    global class_number
    class_number = len(LABELS)
    training_data = []

    def read_database(self, gesture_name):
        dataset = []
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/"
        #gest_sub_dir = "E:/AUTOMATIC_JUN_15th_400_class/AUTOMATIC_JUN_15th_400/"
        #gest_sub_dir = "D:/SORTING60FRAMESDATA/data_reduced/"
        #gest_sub_dir="E:/SORTING_FOLDERS/first_sorted_and_related_fusion/sorting_gestures_with55frames/"
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/raw_data_26to55/"
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/raw26_50/"
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/forteen_signs/"
        #gest_sub_dir = "E:/raw_data_june_18th/automati_luciana/"
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/raw_data_25to55/"
        #gest_sub_dir = "D:/RAW_DATA_TEST_FOR_CONFERENCE/raw_data_24_48/"
        #est_sub_dir = "D:/July_18 data/automati_luciana/raw_data24-48sample600/"
        #gest_sub_dir = "D:/July_18 data/automati_luciana/"
        #gest_sub_dir = "C:/Users/Eric/Desktop/Frame_Range_27-43/"
        gest_sub_dir = "C:/Users/PC_User/PycharmProjects/mmwave/mmwave/Auto_cut_data_collection_from_mmradar/cut_frame_save_path/only_ges_last_all/"
        #gest_sub_dir = "C:/Users/Eric/Desktop/FOLDER41_42/"
        gesture_path = gest_sub_dir + gesture_name
        lenn  = len(os.listdir(gesture_path))


        for gest_num in range(0,lenn):#gestureの1とか2とか
            each_gesture_path = gesture_path + "/" + str(gest_num+1)

            activity_seq_len = len(os.listdir(each_gesture_path))#フレームの長さ（今見ているジェスチャーの

            FrameNumber = 0
            gestureNum = 0
            framelength = 65
            pointlength = 1150
            maxopoints=1150
            gesturedata = np.zeros((framelength, 4, pointlength))
            if activity_seq_len > framelength:
                continue

            for json_num in range(0, activity_seq_len):
                each_frame_path = each_gesture_path + '/' + str(json_num + 1) + '.json'

                if not os.path.exists(each_frame_path):
                    continue

                with open(each_frame_path, 'r') as load_path:
                    load_dict = json.load(load_path)
                    PointCloud = load_dict['PointCloud']
                    pcBufPing = np.array(PointCloud)

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
                        if (x_pos[i] <= 1) and (x_pos[i] >= -1):
                            if (z_pos[i] <= 1.5) and (z_pos[i] >= -1):
                                if (y_pos[i] <= 2.0):
                                    x[i] = x_pos[i]
                                    y[i] = y_pos[i]
                                    z[i] = z_pos[i]
                                    doppler[i] = doppler_pos[i]

                    framedata = np.array([x, y, z,doppler])
                    gesturedata[FrameNumber] = framedata
                    FrameNumber += 1

            dataset.append(gesturedata)

        dataset = np.array(dataset)
        number_of_samples = len(dataset)
        dataset.shape
        print(dataset.shape)

        return dataset, number_of_samples

    def load_data(self):
        total = 0
        for label in self.LABELS:
            trainset, number_of_samples = self.read_database(label)

            for data in trainset: #ここでラベリングをしている
                self.training_data.append(
                    [np.array(data), np.array([self.LABELS[label]])])  # save data and assign label

            total = total + number_of_samples
            print(label, number_of_samples)

        print("Total number:", total)

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)



framelength = 65
pointlength = 1150

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
        self.lstm = nn.LSTM(pointlength * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num*framelength, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


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

def start_ml(layer, nodes_number, epochs, optimizer_step_value, VAL_PCT, parameters, break_limit):
    global neurons_num
    global num_layers
    global frame_parameters

    num_layers = layer
    neurons_num = nodes_number
    frame_parameters = parameters

    net = LSTM_LIN().to(device)
    gestures = Gestures()
    gestures.load_data()

    # Load data from .npy
    dataset = np.load("training_data.npy", allow_pickle=True)

    # Convert from numpy to torch
    X = torch.Tensor([i[0] for i in dataset])
    y = torch.Tensor([i[1] for i in dataset])

    # Divide dataset to training set and test set
    val_size = int(len(dataset) * VAL_PCT)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    print("Trainingset:", len(train_X))
    print("Testset", len(test_X))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=optimizer_step_value)

    loss_list = []
    accuracy_list = []
    iteration_list = []
    tot_accuracy = 0
    count = 0

    print("Layers: ", layer, "Neurons: ", neurons_num)
    for epoch in range(epochs):  # 3 full passes over the data
        for i in range(len(train_X)):

            data = train_X[i].to(device=device)
            targets = train_y[i].to(device=device)
            targets = torch.tensor(targets, dtype=torch.long)
            #print("data shape,before passing to view and LSTM", data.shape)

            scores = net(data.view(-1, framelength, pointlength*frame_parameters))
            #print("the score /score shape:", scores, scores.shape)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()
            count += 1

            tot_number = 0
            tot_correct = 0
            # Every 10 iterations calculate accuracy
            #print("count % 200", count % 200)
            if count % 200 == 0:

                with torch.no_grad():

                    # Calculate accuracy for each gesture and total accuracy
                    for label in Gestures.LABELS:
                        correct = 0
                        number = 0
                        for a in range(len(test_X)):
                            if test_y[a] == Gestures.LABELS[label]:
                                X = test_X[a].to(device=device)
                                y = test_y[a].to(device=device)

                                output = net(X.view(-1, framelength, pointlength*frame_parameters))
                                # print("output is",output)
                                for idx, i in enumerate(output):
                                    #print("output,idx, i, torch.argmax(i) ", output, idx, i, torch.argmax(i))
                                    if torch.argmax(i) == y[idx]:
                                        tot_correct += 1
                                        correct += 1
                                    number += 1
                                    tot_number += 1

                    tot_accuracy = round(tot_correct / tot_number, 3)

                loss = loss.data
                loss = loss.cpu()
                loss_list.append(loss)
                iteration_list.append(count)
                accuracy_list.append(tot_accuracy)

        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss, tot_accuracy * 100))
        if tot_accuracy * 100 > break_limit:
            break

    with torch.no_grad():
        # Calculate accuracy for each gesture and total accuracy　最終的な精度
        tot_number = 0
        tot_correct = 0
        tot_propability = np.zeros([class_number, class_number + 1])  # propability for every output
        tot_classified = np.zeros([class_number, class_number + 1])  # Predicted labels matrix
        for label in Gestures.LABELS:
            correct = 0
            number = 0
            for a in range(len(test_X)):
                if test_y[a] == Gestures.LABELS[label]:
                    X = test_X[a].to(device=device)
                    y = test_y[a].to(device=device)
                    output = net(X.view(-1,framelength, pointlength* frame_parameters))  # Output

                    # print("output shape",output.shape) #Eric own's comment
                    # print(output)# Eric comment
                    # Calculate propability distribution for each gesture
                    out = F.softmax(output, dim=1)
                    # copy memory back to CPU from GPU
                    out = out.cpu()
                    prob = out.numpy() * 100
                    # print(prob, prob.shape)
                    prob = np.append(prob, 0)

                    y = y.cpu()
                    it = int(y.numpy())
                    tot_propability[it] = tot_propability[it] + prob
                    # last cell is the number of samples
                    tot_propability[it][class_number] = tot_propability[it][class_number] + 1

                    for idx, i in enumerate(output):
                        gesture_label_num = torch.argmax(i)
                        tot_classified[it][gesture_label_num] += 1
                        if torch.argmax(i) == y[idx]:
                            tot_correct += 1
                            correct += 1
                        number += 1
                        #print(number)
                        tot_number += 1

            print(label + " accuracy: ", round(correct / number, 3) * 100)
        tot_accuracy = round(tot_correct / tot_number, 3) * 100
        print("Total accuracy: ", tot_accuracy)
        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:16.1f}'.format})

        # save results
        with open("results.txt", 'a') as f:
            print(tot_accuracy, file=f)

        for i in range(class_number):
            # Saving confusion matrix

            print("Gesture", i)
            gest_num = int(tot_propability[i][class_number])
            tot_propability[i] = np.round((tot_propability[i] / gest_num), 3)
            # print(tot_propability[i], gest_num)
            print(tot_classified[i], gest_num)
        # print("save txt")
        np.savetxt("propability matrix.txt", tot_propability, fmt='%f')
        np.savetxt("confusion matrix.txt", tot_classified, fmt='%f')

        torch.save(net.state_dict(), "../model_list/classification_LSTM_LIN_lastall_65.pth")

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
epochs = 30  # number of epochs
optimizer_step_value = 0.0001  # optimizer step value for pytorch
test_percent = 0.1  # what percentage of the database will be used for the set test
parameters = 4
# parameters = 3
break_limit = 98
layer = 3

nodes_number =64
start_ml(layer, nodes_number, epochs, optimizer_step_value, test_percent, parameters, break_limit)

end_time=time.process_time()
execution_time=end_time-start_time
print("Execution time:",execution_time,"seconds")

