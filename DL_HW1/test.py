from model import * 
from prepro import *


if __name__ == "__main__":
    model=pickle.load(open("model.p",'rb'))
    x = np.array([[3,1,50,0,0,512]])
    y = np.array([[1,0]])
    loss, acc, output=model.test(x,y)
    print(output)
    #[1,0,30,0,0,70] [[0.05646121 0.94353879]]
    # [1,0,10,0,0,70] [[0.05626525 0.94373475]]