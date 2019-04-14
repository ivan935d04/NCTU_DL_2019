from prepro import *
from model_2 import *
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

    num_of_batch = int(np.floor(len(train_data) / N))
    num_of_test_batch = int(np.floor(len(test_data) / N))

    Check = train_data[0:N]
    X, Y =prepro(Check)
    check_f(X,Y,model)

    smooth = 0.3
    step = 0
    epoch = 1000
    count = 0
    best_acc = 0
    best_lr = 0
    lr = 0.01
    model = mod1(N, D, H1, H2)
    step=0
    for u in range(epoch):
        random.shuffle(train_data)
        random.shuffle(test_data)
        for i in range(num_of_batch):
            Train = train_data[i*N:(i+1)*N]
            X, Y=prepro(Train)
            slr = lr
            if step > 100:
                pass
                if step % 1000 ==0:
                    # slr =slr * 0.96
                    pass
            _, _,_ = model.train(X, Y,lr= slr)
            
            if (step % 50== 0):
                X_t,Y_t =prepro(train_data)
                loss_train, acc_train,_ =model.test(X_t,Y_t)
                train += [loss_train]
                train_acc += [1-acc_train]
                

            if (step % 500 == 0):
                X_test,Y_test =prepro(test_data)
                loss_test,acc_test,test_=model.test(X_test, Y_test)
                test_acc += [1-acc_test]
                test+=[loss_test]
                print("step: {} test_loss:{} test_acc:{}".format(step,loss_test,acc_test))
            step+=1
        
    

    
    # train=smooth_f(train,smooth)
    # train_acc=smooth_f(train_acc,smooth)
    # test=smooth_f(test,smooth)
    # test_acc=smooth_f(test_acc,smooth)
    # # val_acc = smooth_f(val_acc,smooth)
    # # val = smooth_f(val,smooth)
    

    plt.plot(range(0,step,50),train,label="train")
    # plt.scatter(range(0,step,500),test,label="test")
    # pickle.dump((range(0,step,50),train),open("first_question_train.p",'wb'))
    plt.xlabel("epoch")
    plt.xticks(list(range(0,100*epoch,50000)),list(range(0,epoch,500)))
    plt.legend(loc="upper left")
    plt.title("training loss")
    plt.show()

    plt.plot(range(0,step,500),test,label="test")
    # pickle.dump((range(0,step,500),test),open("first_question_test.p",'wb'))
    plt.xlabel("epoch")
    plt.xticks(list(range(0,100*epoch,50000)),list(range(0,epoch,500)))
    plt.legend(loc="upper left")
    # plt.yscale("log")
    plt.title("testing loss")
    plt.show()

    plt.plot(range(0,step,50),train_acc,label="train")
    plt.xticks(list(range(0,100*epoch,50000)),list(range(0,epoch,500)))
    plt.title("training error")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(range(0,step,500),test_acc,label="test")
    plt.xticks(list(range(0,100*epoch,50000)),list(range(0,epoch,500)))
    # pickle.dump((range(0,step,50),train_acc),open("first_question_acc_train.p",'wb'))
    # pickle.dump((range(0,step,500),test_acc),open("first_question_acc_test.p",'wb'))

    plt.xlabel("epoch")
    # plt.plot(range(0,step,1000),val_acc,label="val");
    # plt.legend(loc="upper left")
    plt.title("testing error")
    plt.show()

    # pickle.dump(model,open("model.p",'wb'))
    
