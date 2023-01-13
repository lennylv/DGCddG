import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from sklearn.metrics import mean_squared_error
import scipy
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = 'Times New Roman'

def draw_dot(dataset):
    x = np.load('./set2_y.npy')[:, -1].astype(float)
    y = np.load('./' + dataset + '_multiple_o.npy').astype(float)
    # plt.xlim(xmax=12.5, xmin=-7)
    # plt.ylim(ymax=12.5, ymin=-6)
    plt.xlim(xmax=8.5, xmin=-5.5)
    plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-6.5, 12, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.3*x1 + 0.88
    plt.scatter(x, y, alpha=0.3, s=7)
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)
    plt.text(x=-5.0,y=7,s='y = 0.3*x + 0.88')
##    plt.text(x=-5,y=8,s='DGCddG')

    plt.text(x=-5.0,y=5,s='σ = 1.44kcal/mol')
    plt.text(x=-5.0,y=6,s='PCC = 0.281')

#    # plt.text(x=4.0,y=11,s='Detail comparison in Table2')
    plt.text(x=0,y=8,s='SSIPE')
    plt.text(x=0,y=7,s='PCC = 0.24')
    plt.text(x=0,y=6,s='σ = 1.49kcal/mol')
#
#
#    # plt.text(x=4.5,y=6,s='BindprofX  R=0.15  σ = 1.99kcal/mol')
    plt.text(x=4.5,y=8,s='BindprofX')
    plt.text(x=4.5,y=7,s='PCC = 0.15')
    plt.text(x=4.5,y=6,s='σ = 1.99kcal/mol')

    plt.text(x=-5.0,y=8,s='DGCddG')
    # plt.text(x=-5.5,y=9.5,s='PCC = 0.61')
    
    plt.savefig('./'+ dataset +'_multiple.png',dpi=1080)

draw_dot('testset2')
def draw_dot_1(dataset):
    x = np.load('./set1_y.npy')[:, -1].astype(float)
    y = np.load('./testset1.npy').astype(float)
    from scipy import stats
    from sklearn.metrics import mean_squared_error

    
    multiple = np.load('./multiple_idx.npy')
    single = [i for i in range(len(x)) if i not in multiple]

    x = x[multiple]
    y = y[multiple]
    print( stats.pearsonr(x, y) )
    print( np.sqrt(mean_squared_error(x, y)) )
    plt.xlim(xmax=12.5, xmin=-7)
    plt.ylim(ymax=12.5, ymin=-6)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-6.5, 12, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.3*x1 + 0.91
    plt.scatter(x, y, alpha=0.3, s=7)
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)
    plt.text(x=-5.0,y=10,s='y = 0.3*x + 0.88')
##    plt.text(x=-5,y=8,s='DGCddG')

    plt.text(x=-5.0,y=8,s='σ = 2.44kcal/mol')
    plt.text(x=-5.0,y=9,s='PCC = 0.667')

    plt.savefig('./'+ dataset +'_multiple.eps',format='eps')
# draw_dot_1('testset1')

def draw_ace():
    x = np.load('./ace-418/itemlist.npy')[:, -1].astype(float)
    y = -np.load('./ace-418/ace2.npy').astype(float)

    plt.xlim(xmax=5, xmin=-5)
    plt.ylim(ymax=5, ymin=-10)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-6.5, 6, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.50*x1 + 0.88
    plt.scatter(x, y, alpha=0.3, s=15)

    plt.text(x=2.0,y=-8,s='σ = 1.76kcal/mol')
    plt.text(x=2.0,y=-7,s='R = 0.35')
    # plt.show()
    plt.savefig('./ace-418/ace-418.png',dpi=1080)

# draw_ace()

def draw_deep_sars():
    x = np.load('./spike-540/itemlist.npy')[:, -1].astype(float)
    y = -np.load('./spike-540/deep_sars2.npy').astype(float)

    from scipy import stats
    from sklearn.metrics import mean_squared_error

    print( stats.pearsonr(x, y) )
    print( np.sqrt(mean_squared_error(x, y)) )

    plt.xlim(xmax=1, xmin=-6)
    plt.ylim(ymax=4, ymin=-6)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-1.5, 5, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.50*x1 + 0.88
    plt.scatter(x, y, alpha=0.3, s=15)

    plt.text(x=-1,y=-5,s='σ = 1.37kcal/mol')
    plt.text(x=-1,y=-4,s='R = 0.255')
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)
    # plt.show()
    plt.savefig('./spike-540/spike-540.png',dpi=1080)

# draw_deep_sars()

def draw_s645():
    x = np.load('./s645/itemlist.npy')[:, -1].astype(float)
    y = np.load('./s645/ab645_y_pred.npy').astype(float)

    from scipy import stats
    from sklearn.metrics import mean_squared_error

    print( stats.pearsonr(x, y) )
    print( np.sqrt(mean_squared_error(x, y)) )

    plt.xlim(xmax=5, xmin=-1.5)
    plt.ylim(ymax=5, ymin=-1.5)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-1.5, 5, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.1*x1 + 0.88
    plt.scatter(x, y, alpha=0.3, s=15)
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)

    plt.text(x=3,y=-1,s='σ = 1.9kcal/mol')
    plt.text(x=3,y=-0.5,s='R = 0.29')
    # plt.show()
    plt.savefig('./s645/s645.png',dpi=1080)
# draw_s645()

def draw_s4169():
    x = np.load('./s4169/itemlist.npy')[:, -1].astype(float)
    y = np.load('./s4169/s4169_y_pred.npy').astype(float)

    from scipy import stats
    from sklearn.metrics import mean_squared_error

    print( stats.pearsonr(x, y) )
    print( np.sqrt(mean_squared_error(x, y)) )

    plt.xlim(xmax=10, xmin=-10)
    plt.ylim(ymax=6, ymin=-6)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-8, 8, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.7*x1 - 0.5
    plt.scatter(x, y, alpha=0.3, s=15)
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)

    plt.text(x=5,y=-5,s='σ = 1.61kcal/mol')
    plt.text(x=5,y=-4,s='R = 0.416')
    # plt.show()
    plt.savefig('./s4169/s4169.png',dpi=1080)

# draw_s4169()

def draw_s8338():
    x = np.load('./s8338/itemlist.npy')[:, -1].astype(float)
    y = np.load('./s8338/s8338_y_pred.npy').astype(float)

    from scipy import stats
    from sklearn.metrics import mean_squared_error

    print( stats.pearsonr(x, y) )
    print( np.sqrt(mean_squared_error(x, y)) )

    plt.xlim(xmax=10, xmin=-10)
    plt.ylim(ymax=6, ymin=-6)
    # plt.xlim(xmax=8.5, xmin=-5.5)
    # plt.ylim(ymax=9, ymin=-5)
    plt.xlabel('Experiment ΔΔG(kcal/mol)')
    plt.ylabel('Preds ΔΔG(kcal/mol)')

    x1 = np.linspace(-8, 8, 100)
    # y1 = 0.3*x1 + 0.88
    y1 = 0.7*x1 - 0.5
    plt.scatter(x, y, alpha=0.3, s=15)
    plt.plot(x1, y1,linestyle='--',linewidth=1.0,alpha=0.5)

    plt.text(x=5,y=-5,s='σ = 1.68kcal/mol')
    plt.text(x=5,y=-4,s='R = 0.55')
    # plt.show()
    plt.savefig('./s8338/s8338.png',dpi=1080)

##draw_s8338()
