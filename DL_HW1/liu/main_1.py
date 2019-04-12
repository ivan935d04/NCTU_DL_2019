import numpy as np
from hw1_1_classes  import three_layer_model
import pandas as pd

# データの読み込み
csv_file = "titanic.csv"
oda = pd.read_csv(csv_file)
data_file = "data_list.p"
train_data, test_data,val_data = split(data_file)
show_list_survived(train_data)
show_list_survived(test_data)
network = three_layer_model(input_size=6, hidden_size_first=3, hidden_size_second=3, output_size=2)

iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('dL_dW1', 'dL_db1', 'dL_dW2', 'dL_db2', 'dL_dW3', 'dL_db3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
