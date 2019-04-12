import matplotlib.pyplot as plt
import numpy as np 
import pickle 

pclass_more3_acc_test = pickle.load(open("pclass_more3_acc_test.p","rb"))
pclass_more3_acc_train = pickle.load(open("pclass_more3_acc_train.p","rb"))
pclass_more3_loss_train = pickle.load(open("pclass_more3_loss_train.p","rb"))
pclass_more3_loss_test = pickle.load(open("pclass_more3_loss_test.p","rb"))

second_acc_test = pickle.load(open("second_acc_test.p","rb"))
second_acc_train= pickle.load(open("second_acc_train.p","rb"))
second_loss_test= pickle.load(open("second_loss_test.p","rb"))
second_loss_train= pickle.load(open("second_loss_train.p","rb"))

plt.scatter(second_loss_train[0],second_loss_train[1],color="orange",label = "origin")
plt.scatter(pclass_more3_loss_train[0],pclass_more3_loss_train[1],color="green",label="pclass_onehot")
epoch = 1000
plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
plt.legend(loc="upper left")
plt.xlabel("epoch")
plt.title("train Loss")
plt.show()

plt.plot(second_acc_train[0],second_acc_train[1],color="orange",label = "origin")
plt.plot(pclass_more3_acc_train[0],pclass_more3_acc_train[1],color="green",label="pclass_onehot")
plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
plt.legend(loc="upper left")
plt.xlabel("epoch")
plt.title("train Error Rate")
plt.show()







