import os
import numpy as np
os.system('python cv_fold_645.py --fold 0 --validation 1')
epoch = np.load('best_epoch.npy')[0]
for i in range(5):
    os.system('python cv_fold_645.py --fold ' + str(i) + ' --epoch '+str(epoch))