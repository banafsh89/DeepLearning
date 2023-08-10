###########################
###This is to create the images from the data to look at them using pixelmator and make the masks
########################

import pandas as pd
import numpy as np
np.random.seed(9001)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.set_printoptions(precision=16)
import os, glob


def RecustImage():

    pd.set_option('display.precisio',18)
    np.set_printoptions(precision=18)

    
    filenames=sorted(glob.glob(os.path.join("/work/banafsheh.beheshtipour/CleanDeepLearning/paper2/Inputfiles/ImageMaskInjData/imageHR/image", 'res*')))
    for filenameI in filenames:
        a=filenameI.split("_")
        f_0=a[-1]
        print(f_0)
        dfpix=pd.read_csv("{}/all_LRes{}.t2".format(filenameI,f_0), sep='\s+', header=None)
        dfpix.columns=['fpix', 'fdotpix','aveBSGLtLr']

        dfIC=pd.read_csv("{}/binedI_{}.t2".format(filenameI,f_0), sep='\s+', header=None)
        dfIC.columns=['f', 'fdot','BSGLtLr']

        dfinj=pd.read_csv("{}/injections{}.txt".format(filenameI,f_0), sep='\s+', header=None)


       
        f_0char=a[1]
        f_0=float(f_0char)
        f_1=f_0+0.05

        
        ##Creat the LR image:
        
        dfpix['aveBSGLtLr']=dfpix['aveBSGLtLr'].replace(0.00001,np.nan)
        mm=dfpix.aveBSGLtLr.max()

        #to make the injection points on the plot (having higher value to be red)
        for f,fdot in zip(dfinj[4].values,dfinj[5].values):
            tt=dfpix[(dfpix['fpix']<(f+2))&(dfpix['fpix']>(f-2))]
            ttt=tt[(tt['fdotpix']<(fdot+2))&(tt['fdotpix']>(fdot-2))]
            ind=ttt.index
            for i in ind:
                dfpix.aveBSGLtLr[i]=mm+5 #to make the value of pixel to be higher so sees as red in the image




        pivottrain=(dfpix.groupby(['fpix','fdotpix'])['aveBSGLtLr'].max()).unstack()
        pivottrain.index.name = None
        pivottrain.columns.name = None
        arrt=pivottrain.values

        # a colormap and a normalization instance
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=arrt.min(), vmax=arrt.max())
        plt.imsave('{}/imageLR_{}_{}.png'.format(filenameI,f_0char,dfinj.shape[0]), arrt, cmap=cmap)


        ##Create the HR image:
        pivottrain=(dfIC.groupby(['f','fdot'])['BSGLtLr'].max()).unstack()
        pivottrain.index.name = None
        pivottrain.columns.name = None
        arrt=pivottrain.values

        # a colormap and a normalization instance
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=arrt.min(), vmax=arrt.max())
        plt.imsave('{}/OrIimageHR_{}.png'.format(filenameI,f_0char), arrt, cmap=cmap)


        for f,fdot in zip(dfinj[2].values,dfinj[3].values):

            #to make the injection points on the plot (having higher value to be red)
            #print(f,fdot)
            dd=30 ##to make a larger area to be red so one can easily see it in the images


            ff=list(range(f-dd,f+dd))
            ffdot=list(range(fdot-dd,fdot+dd))
            bs=[0.00001]*dd*2

            d={'f':ff,'fdot':ffdot,"BSGLtLr":bs}
            pf=pd.DataFrame(data=d)
            dfIC=dfIC.append(pf, ignore_index=True)

            tt=dfIC[(dfIC['f']<(f+dd))&(dfIC['f']>(f-dd))]
            ttt=tt[(tt['fdot']<(fdot+dd))&(tt['fdot']>(fdot-dd))]
            mm=dfIC.BSGLtLr.max()

            ind=ttt.index
            for i in ind:
                dfIC.BSGLtLr[i]=mm+5 #to make the value of pixel to be higher so sees as red in the image



        pivottrain=(dfIC.groupby(['f','fdot'])['BSGLtLr'].max()).unstack()
        pivottrain.index.name = None
        pivottrain.columns.name = None
        arrt=pivottrain.values

        # a colormap and a normalization instance
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=arrt.min(), vmax=arrt.max())
        plt.imsave('{}/imageHR_{}.png'.format(filenameI,f_0char), arrt, cmap=cmap)


    print ("Done")

if __name__ == "__main__":
    
    RecustImage()

