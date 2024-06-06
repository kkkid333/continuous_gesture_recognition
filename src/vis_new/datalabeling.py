import json


jesunum = 3
lis = []
lis_min = []


for renzoku_num in range(jesunum):
    for cut in range(2):
        s = input()
        lis_min.append(s)
    lis.append(lis_min)
    lis_min = []

#print(lis)

with open("C:/Users/PC_User/PycharmProjects/mmwave/mmwave/vis_new/correct_data/swipe_left_swipe_down_tap/1.json","w") as f:
    json.dump(lis,f)
