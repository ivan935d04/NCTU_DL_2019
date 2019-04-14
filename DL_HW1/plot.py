import matplotlib.pyplot as plt
import numpy as np 
import pickle 

first_question_acc_test = pickle.load(open("first_question_acc_test.p","rb"))
first_question_acc_train = pickle.load(open("first_question_acc_train.p","rb"))
first_question_loss_train = pickle.load(open("first_question_loss_train.p","rb"))
first_question_loss_test = pickle.load(open("first_question_loss_test.p","rb"))

second_acc_test = pickle.load(open("second_acc_test.p","rb"))
second_acc_train= pickle.load(open("second_acc_train.p","rb"))
second_loss_test= pickle.load(open("second_loss_test.p","rb"))
second_loss_train= pickle.load(open("second_loss_train.p","rb"))

plt.scatter(second_loss_test[0],second_loss_test[1],color="orange",label = "origin")
plt.scatter(first_question_loss_test[0],first_question_loss_test[1],color="green",label="first_question")
epoch = 1000
plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
plt.legend(loc="upper left")
plt.xlabel("epoch")
plt.title("test Loss")
plt.show()

plt.plot(second_acc_test[0],second_acc_test[1],color="orange",label = "origin")
plt.plot(first_question_acc_test[0],first_question_acc_test[1],color="green",label="first_question")
plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
plt.legend(loc="upper left")
plt.xlabel("epoch")
plt.title("test Error Rate")
plt.show()







