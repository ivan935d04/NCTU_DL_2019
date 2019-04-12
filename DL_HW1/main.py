from prepro import *
from model import *
import random
import matplotlib.pyplot as plt
from gradient_check import eval_numerical_gradient
import pickle

csv_file = "titanic.csv"
oda = pd.read_csv(csv_file)
data_file = "data_list.p"

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def smooth_f(data, smooth=0.3):
    for i in range(len(data)):
        if i == 0:
            temp = data[i];
            continue
        else:
            data[i] = data[i] * smooth + temp * (1-smooth)
            temp = data[i]
    return data

def check_f(X,Y,net):
    
    loss, grads = net.loss_f(X, Y)
    
    for param_name in grads:
        f = lambda W: net.loss_f(X, Y)[0]

        param_grad_num = eval_numerical_gradient(f, net.param[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

if __name__ == "__main__":

    N, D, H1, H2 = 8, 6, 3, 3
    print("batch_size:{}, h1 size:{} h2 size {}".format(N,H1,H2))
    model = mod1(N, D, H1, H2)
    train_data, test_data= split(data_file)
    show_list_survived(train_data)
    show_list_survived(test_data)


    
    step = 0

    test = []
    test_acc = []
    test_t = []
    test_acc_t = []

    train = []
    train_acc = []

    val = []
    val_t = []
    val_acc = []
    val_acc_t = []

    # train_number = list(range(len(train_data)))
    # test_number = list(range(len(test_data)))
    print(len(train_data))
    num_of_batch = int(np.floor(len(train_data) / N))
    print(num_of_batch)
    num_of_test_batch = int(np.floor(len(test_data) / N))

    Check = train_data[0:N]
    X, Y =prepro(Check)
    check_f(X,Y,model)

    smooth = 0.3
    lr= 1e-2
    step = 0
    epoch = 1000
    count = 0
    print("lr: {}".format(lr))
    # print(len(train_data))
    for u in range(epoch):
        # random.shuffle(train_data)
        for i in range(num_of_batch):
            Train = train_data[i*N:(i+1)*N]
            X, Y=prepro(Train)
            if step > 100:
                if step % 1000 ==0:
                    lr = lr * 0.96
                    print("lr={}".format(lr))  
            loss_train, _,_ = model.train(X, Y,lr= lr)
            
            if (step % 50== 0):
                train += [loss_train]
                X_t,Y_t =prepro(train_data)
                _, acc_train,_ =model.test(X_t,Y_t)
                # print(model.grad)
                print("------------------------------------")
                train_acc += [1-acc_train]
                print("step: {} train_loss:{} train_acc:{}".format(step,loss_train,acc_train))

            if (step % 500 == 0):
                for k in range(num_of_test_batch):
                    Test =test_data[k*N:(k+1)*N]
                    x_test, y_test = prepro(Test)
                    loss_test,_,_ = model.test(x_test, y_test)
                    test_t += [loss_test]
                X_test,Y_test =prepro(test_data)
                _,acc_test,test_=model.test(X_test, Y_test)
                test_acc += [1-acc_test]
                test += [np.mean(test_t)]
                print("step: {} test_loss:{} test_acc:{}".format(step,np.mean(test_t),acc_test))
                test_t = []
                test_acc_t = []
            step+=1
            # if step >1000
    

    
    # train=smooth_f(train,smooth)
    # train_acc=smooth_f(train_acc,smooth)
    # test=smooth_f(test,smooth)
    # test_acc=smooth_f(test_acc,smooth)
    # # val_acc = smooth_f(val_acc,smooth)
    # # val = smooth_f(val,smooth)
    

    plt.plot(range(0,step,50),train,label="train")
    # plt.scatter(range(0,step,500),test,label="test")
    # pickle.dump((range(0,step,50),train),open("pclass_more3_train.p",'wb'))
    plt.xlabel("epoch")
    plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
    plt.legend(loc="upper left")
    plt.title("training loss")
    plt.show()

    plt.plot(range(0,step,500),test,label="test")
    # pickle.dump((range(0,step,500),test),open("pclass_more3_test.p",'wb'))
    plt.xlabel("epoch")
    plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
    plt.legend(loc="upper left")
    # plt.yscale("log")
    plt.title("testing loss")
    plt.show()


    # plt.subplot(212)
    # plt.ylim((0.5,0.85))
    # plt.yscale("log")
    # plt.xscale("log")
    plt.plot(range(0,step,50),train_acc,label="train")
    plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
    plt.title("training error")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(range(0,step,500),test_acc,label="test")
    plt.xticks(list(range(0,100*epoch,10000)),list(range(0,epoch,100)))
    # pickle.dump((range(0,step,50),train_acc),open("pclass_more3_acc_train.p",'wb'))
    # pickle.dump((range(0,step,500),test_acc),open("pclass_more3_acc_test.p",'wb'))

    plt.xlabel("epoch")
    # plt.plot(range(0,step,1000),val_acc,label="val");
    # plt.legend(loc="upper left")
    plt.title("testing error")
    plt.show()

    # pickle.dump(model,open("model.p",'wb'))
    
