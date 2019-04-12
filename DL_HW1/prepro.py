import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import pickle
import random 
from sklearn import preprocessing

csv_file = "titanic.csv"
oda = pd.read_csv(csv_file)

def show(oda):
    print(oda.columns.values)  #get column information
    for i in range(oda.shape[0]):
        for j in range(oda.shape[1]):
            print(oda.values[i][j], end=' ')
        print("")
def show_survived(oda):
    count = 0
    u_count = 0
    for i in range(oda.shape[0]):

        if oda.values[i][0] == 0:
            count +=1
        if oda.values[i][1] == 1:
            u_count +=1
    print("Unsurvived: {}, Survived: {}".format(u_count,count))
def show_list_survived(list_):
    u_count = 0
    count = 0
    for i in range(len(list_)):
        if list_[i]['y'] == 0:
            u_count += 1
        else:
            count += 1
    print('Survived: {} UnSurvived: {}'.format(count, u_count))

def standardize(array):
    std=np.std(array)
    mean=np.mean(array)
    output=(array - mean)/std
    return output

def show_relationship(oda):
    plt.scatter(oda[:,3][(oda[:,0]==0) & (oda[:,2]==1)], oda[:,0][(oda[:,0]==0) & (oda[:,2]==1)], color="red",label="death")
    plt.scatter(oda[:,3][(oda[:,0]==1) & (oda[:,2]==0)], oda[:,0][(oda[:,0]==1) & (oda[:,2]==0)], color="blue",label="survived")
    
    plt.show()



def store(oda):
    data = oda.values
    d_l = list()
    dic_t = {"feat": None, "y": None}
    data[:,6]=standardize(data[:,6])
    for i in range(data.shape[0]):
        y = data[i][0]
        x = data[i][1:]
        dic_t['feat'] = x
        dic_t['y'] = y
        d_l.append(copy.deepcopy(dic_t))
    pickle.dump(d_l, open("data_normalize_list.p", "wb"))


def split(file_name):
    data = pickle.load(open(file_name, "rb"))
    # random.shuffle(data)
    test = 800
    
    test_data = data[test:]
    print(len(test_data))
    train_data   = data[:test]
    # val_data  = train_data[-91:]
    # train_data = train_data[:
    return  train_data,test_data

def prepro(sheets):
    x_temp=np.zeros((len(sheets),sheets[1]['feat'].shape[0]))
    y_temp=np.zeros((len(sheets),2))

    for i in range(len(sheets)):
        x_temp[i] = sheets[i]['feat']
        # if x_temp[i][2] == 1:
        #     x_temp[i][:2] = 0
        # elif x_temp[i][2] == 2:
        #     x_temp[i][:3] = np.array([0,1,0])
        # elif x_temp[i][2] == 3:
        #     x_temp[i][:3] = np.array([1,0,0])
        # x_temp[i][2] = 0
        if sheets[i]["y"] == 1:
            y_temp[i][1] = 1
        else:
            y_temp[i][0] = 1
    # np.random.shuffle(x_temp)
    # np.random.shuffle(y_temp)
    return x_temp,y_temp

if __name__ == "__main__":
    # store(oda)
    # show_survived(oda)
    # show_relationship(oda.values)
    print(oda.describe())
