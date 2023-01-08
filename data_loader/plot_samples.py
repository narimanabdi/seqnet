import matplotlib.pyplot as plt
import numpy as np

def plot_test(Xs,Xq,names,predicted_names,size=5,font_size=5):
    D = Xs[:,0]
    fig, axes = plt.subplots(nrows=2,ncols=len(D),
            figsize=(size,size),subplot_kw={'xticks': [], 'yticks': []})
    for _ in range(len(Xs)):
        for n in range(len(D)):
            axes[0][n].imshow(D[n]/255)
            axes[0][n].set_title(names[n],c="green",size=font_size)
            axes[1][n].imshow(Xq[n][0]/255)
            if predicted_names[n] == names[n]:
                color = 'green'
            else:
                color = 'red'
            axes[1][n].set_title(predicted_names[n],c=color,size=font_size)
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.9)
    plt.show()

def plot_test_multi_line(Xs,Xq,names,predicted_names,size=5,font_size=5,hspace=0.0,col_num=10):
    D = Xs[:,0]
    r = int(np.floor(D.shape[0] / col_num))
    fig, axes = plt.subplots(nrows=r*2,ncols=col_num,
            figsize=(size,size),subplot_kw={'xticks': [], 'yticks': []})
    n = 0
    for row in range(0,2*r,2):
        for col in range(col_num):
            axes[row][col].imshow(D[n]/255)
            axes[row][col].set_title(names[n],c="green",size=font_size)
            axes[row+1][col].imshow(Xq[n][0]/255)
            if predicted_names[n] == names[n]:
                color = 'green'
            else:
                color = 'red'
            axes[row+1][col].set_title(predicted_names[n],c=color,size=font_size)
            n += 1
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    plt.show()