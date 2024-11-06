from nilearn.image import mean_img, load_img, concat_imgs,index_img,math_img,clean_img, resample_to_img, new_img_like
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from itertools import product
import nilearn.decoding 
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import sys
from shapely.geometry import Polygon
from sklearn.decomposition import PCA


class cv_circularity(BaseEstimator):
    def __init__(self, parameter=1):
        self.parameter = parameter

    def fit(self, X, y):
        """
        Fit the estimator to the training data.
        Note the range of y_train starts from 1 and should be consecutive integers
        """
        self.classes_ = sorted(set(y))
        if np.min(self.classes_)!=1:
            sys.exit("The smallest label should be 1!")    
        y=np.array(y)
        np.random.shuffle(y) #shuffle the trial label

        #in case one run doesnt have all the conditions skip this fold
        #change to 9 if stimulus space
        if np.unique(y).shape[0]!=4:
            self.ci=np.nan
        else:
            train_data = np.zeros((np.unique(y).shape[0], X.shape[1]))
            for i in range(np.unique(y).shape[0]):
                train_data[i,:] = np.mean(X[y==i+1],axis=0)
            
            pca1 =  PCA(n_components=2)
            scaler = StandardScaler()
            coor = pca1.fit_transform(scaler.fit_transform(train_data))
            #####for stimulus space########
            # coor = pca1.fit_transform(scaler.fit_transform(np.delete(train_data, [1,4,7],axis=0)))
            c1x =coor[:,0]
            c1y =coor[:,1]
            polygon = Polygon(((c1x[0],c1y[0]),(c1x[1],c1y[1]),(c1x[3],c1y[3]),(c1x[2],c1y[2]),(c1x[0],c1y[0])))
            #####for stimulus space########
            # polygon = Polygon(((c1x[0],c1y[0]),(c1x[1],c1y[1]),(c1x[3],c1y[3]),(c1x[5],c1y[5]),(c1x[4],c1y[4]),(c1x[2],c1y[2]),(c1x[0],c1y[0])))
            self.ci = 4*np.pi*polygon.area/(polygon.length)**2
        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self

    #empty function to keep compatible with Nilearn requirement for customized decoder
    def score(self, X, y):
        a=0
        return self.ci

for sub in np.arange(1,22):
    #a gray matter mask converted from HCP atlas
    mask_fname=  'HCP_GM_mask.nii.gz' 

    # Load behavioral information
    beha_fname = glob.glob('s0{a}_behavioural.csv'.format(a=sub))
    print(beha_fname)
    behavioural = pd.read_csv(beha_fname[0], delimiter=',',encoding_errors='ignore')
    behavioural['size_num'] = np.where(np.logical_or(behavioural.cue.values==1,behavioural.cue.values==2),0, 1) # bigger=0, smaller=1
    behavioural['color_num'] = np.where(np.logical_or(behavioural.cue.values==1,behavioural.cue.values==3),0, 1) # red=0, greener=1
    for j,l in enumerate(list(product((1,2,3),(1,2,3)))): #for testing stimulus space
        behavioural.loc[(behavioural['whichSize']==l[0])&(behavioural['whichColor']==l[1]), 'trial_type'] = j+1

    print('num of trial entering searchlight: ', behavioural.shape[0])
    session_label = behavioural['block_loop.thisN'].values      
    #since we dont really need cross-validation for circularity index, assign one trial to a different group label
    #to fake the group number as 2, bc nilearn searchlight requires a cv instance to be passed
    new_sess_num = np.zeros(session_label.shape[0])
    new_sess_num[0]=1
    ref_img = "s0{a}_delay1_smooth.nii.gz".format(a=sub)

    cv = LeaveOneGroupOut()
    for tr_ in ['delay1','delay2']:
        train_niimgs =load_img('s0%s_%s_smooth.nii.gz'%(sub, tr_))
        train_niimgs_z = clean_img(train_niimgs, runs=session_label, standardize=True, detrend=True) #standardize images
        #generate 50 shuffled images per subject to create null distribution
        for i in range(50):
            #run searchlight analysis
            searchlight = nilearn.decoding.SearchLight( estimator=cv_circularity(),
                                                        mask_img=mask_fname ,
                                                        process_mask_img=None,
                                                        radius=9, n_jobs=20, 
                                                        verbose=0, cv=cv)
            y_train = behavioural.cue.values #for goal space
            # y_train = behavioural.trial_type.values #for stimulus space

            searchlight.fit(train_niimgs_z, y_train,  groups=new_sess_num)
            sl_img = new_img_like(ref_img, searchlight.scores_)
            sl_img.to_filename('s0%s_searchlight_goal_%s_smooth_shuffled_%d.nii.gz'%(sub, tr_, i))
