
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:21:43 2020

@author: 

- set of functions used to create a dataset for neural network training
- user has to create a folder then set its path in 'root'
- use the line 'createDataset(root)' to generate the dataset

"""

root='C:/Users/example_path/DatasetFolder/'

import os
import cv2
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

def createDataset(root):
    """
    Main function used to create the synthetic dataset in the main foler 'root'

    """
    createPwc(root)
    applyGradToPwc(root)
    createDef(root)

def applyField(img):
    """
    Creates a fine structure map and applies it to 'img'. The image array of 
    'img' with added fine structures and the image array of the fine structures
    are both returned.

    """
    
    img_shape=img.shape
    
    # Variables determining the range of possible defects characteristics
    modulation=False
    gainAndLoss=True
    direction=randint(0,3)
    dotRadius=randint(1, 5)
    numberOfSeeds=randint(2, 8)
    #intensityGainMin=randint(1,6)
    intensityGainMin=randint(3,10)
    #intensityGainMax=randint(7,22)
    intensityGainMax=randint(11,30)
    branchOffProbability=randint(0,20)
    branchDeathProbability=randint(0,20)
    
    field=fracField(img_shape,numberOfSeeds,intensityGainMin,intensityGainMax,
                    branchOffProbability,branchDeathProbability,gainAndLoss)
    
    field*=dotField(img_shape,numberOfSeeds,intensityGainMin
                    ,intensityGainMax,dotRadius,gainAndLoss)
    
    #field*=gradField(img_shape, direction, intensityGainMax)
    
    if modulation:
        img=abs(255-img)
        idef=img*field
        
        idef[idef<0]=0
        idef[idef>255]=255
        idef=abs(255-idef)
        
        field=1/field        
        
    else:
        idef=(np.max(img)/2)*field+img-(np.max(img)/2)
    
    return idef,field

def createDef(root):
    """
    Creates folders containing the images of the smooth piecewise (/pwc), fine 
    scrutures (/fra) and their combination (/def).

    """

    setList=['train', 'validation', 'test']

    for s in setList:
        path=root+s+'/'
        if not os.path.isdir(path+'def/'):
            os.mkdir(path+'def/')
        if not os.path.isdir(path+'fra/'):
            os.mkdir(path+'fra')
        for filename in os.listdir(path+'pwc/'):
            ipwc=cv2.imread(path+'pwc/'+filename,0)
            
            # normalization
            ipwc=np.around((ipwc-np.min(ipwc))/np.max(ipwc-np.min(ipwc))*255)
            
            [idef,field]=applyField(ipwc)
            
            cv2.imwrite(path+'pwc/'+filename,ipwc)
            cv2.imwrite(path+'def/'+filename,idef)
            
            #if not gainAndLoss :
            #cv2.imwrite(path+'fra/'+filename,np.around(field*100-100))
            
            # One output for frac:
            #cv2.imwrite(path+'fra/'+filename,np.around(field*100))    
            
            
            #Two channel output for frac:
                
            fieldPos=field.copy()
            fieldNeg=field.copy()
            
            fieldPos[field<1]=1
            fieldNeg[field>1]=1
            
            fieldPos=(fieldPos-1)*100
            fieldNeg=(1-fieldNeg)*100
            
            fieldComp=np.stack((fieldPos,fieldNeg,np.zeros(field.shape)),axis=2)
            
            cv2.imwrite(path+'fra/'+filename,np.around(fieldComp)) 
            

def gradField(img_shape,direction,intensityGainMax):
    """
    Creates an intensity gradient map of shape 'img_shape', intensity 
    'intensityGainMax' with a direction [0,3].

    """
    
    if direction==0:
        x=np.linspace(0, intensityGainMax/100, img_shape[0])
        y=np.ones(img_shape[1])
    if direction==1:
        x=np.linspace(intensityGainMax/100, 0, img_shape[0])
        y=np.ones(img_shape[1])
    if direction==2:
        y=np.linspace(0, intensityGainMax/100, img_shape[1])
        x=np.ones(img_shape[0])
    if direction==3:
        y=np.linspace(intensityGainMax/100, 0, img_shape[1])
        x=np.ones(img_shape[0])
    
    gainOrLoss=1
    
    field=(x[np.newaxis].T*y[np.newaxis]*gainOrLoss+1)
    
    #return np.around(field,decimals=2)
    return field
    

def dotField(img_shape,numberOfSeeds,intensityGainMin,intensityGainMax,
             dotRadius,gainAndLoss):
    """
    Creates a map with shady spot defects

    """
    size=dotRadius*2+1
    sigma=size/4
    
    base_seeds=[]
    for i in range(1,numberOfSeeds):
        seed_x=randint(dotRadius+1, img_shape[0]-dotRadius-1)
        seed_y=randint(dotRadius+1, img_shape[1]-dotRadius-1)
        base_seeds.append([seed_x,seed_y])
    
    field=np.ones(img_shape)
    
    for [seed_x,seed_y] in base_seeds:
        
        gainOrLoss=-1
        
        if gainAndLoss:
            gainOrLoss=randint(0, 1)*2-1    
    
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2))
                                 * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))
                                              /(2*sigma**2)), (size, size))
        kernel-=np.min(kernel)
        kernel=1+(kernel/np.max(kernel))*randint(intensityGainMin,intensityGainMax)/100*gainOrLoss
        field[seed_x-dotRadius:seed_x+dotRadius+1, 
              seed_y-dotRadius:seed_y+dotRadius+1]=np.multiply(field[
                  seed_x-dotRadius:seed_x+dotRadius+1, seed_y-dotRadius:seed_y+dotRadius+1], kernel)
        
    return np.around(field,decimals=2)
                                                       

def fracField(img_shape,numberOfSeeds,intensityGainMin,intensityGainMax,
              branchOffProbability,branchDeathProbability,gainAndLoss):
    """
    Creates a map with fracture/branches defects

    Parameters
    ----------
    img_shape : the shape of the fracture field to be placed on the img
    numberOfSeeds : number of seeds which will initiate a crack 
    intensityGainMin : minimum gain of intensity value in a crack [0;100] 
    intensityGainMax : maximum gain of intensity value in a crack [0;100] 
    branchOffProbability : probability (%) of a seed to create a secondary crack along its path (at each step) 
    branchDeathProbability : probability (%) of a secondary crack to end at each step 


    Returns
    -------
    Returns a intensity change map with values [-1,1]

    """

    field=np.ones(img_shape)
    
    base_seeds=[]
    for i in range(1,numberOfSeeds):
        seed_x=randint(0, img_shape[0])
        seed_y=randint(0, img_shape[1])
        base_seeds.append([seed_x,seed_y])
        
    seeds=[]
    
    for [seed_x,seed_y] in base_seeds:
        rand_dir=randint(0, 7)
        gainOrLoss=-1
        if gainAndLoss :
            gainOrLoss=randint(0, 1)*2-1
        while seed_x < img_shape[0]-1 and seed_x>0 and seed_y < img_shape[1]-1 and seed_y>0:
            field[seed_x,seed_y]=field[seed_x,seed_y]+randint(intensityGainMin,intensityGainMax)/100*gainOrLoss
            [seed_x,seed_y]=nextPixel([seed_x,seed_y], rand_dir)

            if randint(0,100) < branchOffProbability:
                seeds.append([seed_x,seed_y,gainOrLoss])
    
    for [seed_x,seed_y,gainOrLoss] in seeds:
        rand_dir=randint(0, 7)
        gainOrLoss=-1
        if gainAndLoss:
            gainOrLoss=randint(0, 1)*2-1
        while seed_x < img_shape[0]-1 and seed_x>0 and seed_y < img_shape[1]-1 and seed_y>0:
            field[seed_x,seed_y]=field[seed_x,seed_y]+randint(intensityGainMin,intensityGainMax)/100*gainOrLoss
            [seed_x,seed_y]=nextPixel([seed_x,seed_y], rand_dir)
            
            if randint(0,100) < branchDeathProbability:
                break
            
    return field
                
def nextPixel(currentPos,direction):
    """
    Parameters
    ----------
    currentPos : [pos_x,pos_y]
    direction : [0,7]

    Returns
    -------
    Returns new pixel coordinates in the direction

    """

    [seed_x,seed_y]=currentPos

    if direction==0:
        seed_x=seed_x+randint(0,1)
        seed_y=seed_y+randint(0,1)
    if direction==1:
        seed_x=seed_x+randint(0,1)
        seed_y=seed_y-randint(0,1)
    if direction==2:
        seed_x=seed_x-randint(0,1)
        seed_y=seed_y-randint(0,1)
    if direction==3:
        seed_x=seed_x-randint(0,1)
        seed_y=seed_y+randint(0,1)
    if direction==4:
        seed_x=seed_x+1
        seed_y=seed_y+randint(-1,1)
    if direction==5:
        seed_x=seed_x+randint(-1,1)
        seed_y=seed_y+1
    if direction==6:
        seed_x=seed_x-1
        seed_y=seed_y+randint(-1,1)
    if direction==7:
        seed_x=seed_x+randint(-1,1)
        seed_y=seed_y-1
                
    return [seed_x,seed_y]

def createPwc(root):
    """
    Creates a folder with the piecewise smooth component images

    """
    setList=['train', 'validation', 'test']

    for s in setList:
        
        if not os.path.isdir(root+s):
            os.mkdir(root+s)
        
        path=root+s+'/'
        if not os.path.isdir(path+'pwc/'):
            os.mkdir(path+'pwc/')
            
    pieceGen(root)

def pieceGen(root):
    """
    Creates smooth piecewise images and distributes them in different folders
    corresponding to the train, validation and test datasets.
    /!\ The size of the images created depends on the monitor resolution.

    """
    
    max_num_shape = 7
    ds_size = 2000
    [train, validation, test]=[70, 20, 10]
    
    for ind in range(ds_size):
        
        NUM_ells = np.random.randint(2,max_num_shape)
        NUM_rect = np.random.randint(2,max_num_shape)
        ells = [Ellipse(xy=np.random.rand(2) * 10,
                        width=np.random.randint(3,6) , height=np.random.randint(4,8),
                        angle=np.random.rand() * 360) for i in range(NUM_ells)]
        rect = [Rectangle(xy=(np.random.rand(2) * 10),
                          width=np.random.randint(3, 6), height=np.random.randint(4, 8),
                            angle=np.random.randint(0,360)) for i in range(NUM_rect)]
        
        fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=False, dpi= 80)
        #fig.set_size_inches(1.91, 1.91)
        fig.set_size_inches(3.51,3.51)
        for e in ells:
            col = np.random.uniform(0, 1, 1)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(np.random.rand())
            e.set_facecolor(np.repeat(col, 3))
        for m in rect:
            col = np.random.uniform(0, 1, 1);
            ax.add_patch(m)
            m.set_clip_box(ax.bbox)
            m.set_alpha(np.random.rand())
            m.set_facecolor(np.repeat(col, 3))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
    
        #fig.draw()
        ax.margins(0)
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.tick_params(which='both', direction='in')
        
        if ind < np.around(ds_size/100*train):
            output_path = root+'/train/pwc/'
        elif ind >= np.around(ds_size/100*(100-test)):
            output_path = root+'/test/pwc/'
        else:
            output_path = root+'/validation/pwc/'
            
        fig.tight_layout()
        fig.savefig(os.path.join(output_path + str(ind) + '.png'),
                    bbox_inches='tight', pad_inches=0.0, edgecolor='w')
    
        plt.close('all')

def applyGradToPwc(root):
    """
    Applies a intensity gradient to the smooth piecewise images

    """
    
    setList=['train', 'validation', 'test']

    for s in setList:
        path=root+s+'/'
        for filename in os.listdir(path+'pwc/'):
            if randint(0, 1)==1:
                direction=randint(0,3)
                intensityGainMax=randint(7,22)
                ipwc=np.float64(cv2.imread(path+'pwc/'+filename,0))
                img_shape=ipwc.shape
                ipwc*=gradField(img_shape, direction, intensityGainMax)
                ipwc[ipwc<0]=0
                ipwc[ipwc>255]=255
                cv2.imwrite(path+'pwc/'+filename,ipwc)
    
def testDiff(root):
    """
    Test function used to creates images based on the difference between the 
    smooth piecewise images with and without defects
    
    """
    setList=['train', 'validation', 'test']

    for s in setList:
        path=root+s+'/'
        if not os.path.isdir(path+'diff/'):
            os.mkdir(path+'diff/')
        for filename in os.listdir(path+'pwc/'):
            ipwc=cv2.imread(path+'pwc/'+filename,0)
            idef=cv2.imread(path+'def/'+filename,0)
            
            # One channel output for frac
            #idiff=100+idef-ipwc
            
            #cv2.imwrite(path+'diff/'+filename,idiff)
            
            # Two channel output for frac
            
            #idiff=np.zeros(idef.shape,dtype=float)
            idiff=np.asarray(idef,float)-np.asarray(ipwc,float)
            
            fieldPos=idiff.copy()
            fieldNeg=idiff.copy()
            
            fieldPos[idiff<0]=0
            fieldNeg[idiff>0]=0
            
            fieldNeg=abs(fieldNeg)
            
            fieldComp=np.stack((np.zeros(idiff.shape),fieldPos,fieldNeg),axis=2)
            
            # Inverted colors
            
            fieldComp[:,:,0]=255-fieldComp[:,:,0]
            fieldComp[:,:,1]=255-fieldComp[:,:,1]
            fieldComp[:,:,2]=255-fieldComp[:,:,2]
            
            
            cv2.imwrite(path+'diff/'+filename,np.around(fieldComp)) 
            
def testDiffPW(root):
    """
    Test function used to creates images based on the difference between the 
    smooth piecewise images with and without defects
    
    """
    setList=['train', 'validation', 'test']

    for s in setList:
        path=root+s+'/'
        if not os.path.isdir(path+'diffPW/'):
            os.mkdir(path+'diffPW/')
        for filename in os.listdir(path+'pwc/'):
            ipwc=cv2.imread(path+'pwc/'+filename,0)
            idef=cv2.imread(path+'def/'+filename,0)
            
            idiff=np.asarray(idef,float)-np.asarray(ipwc,float)
            
            fieldPos=idiff.copy()
            fieldNeg=idiff.copy()
            
            fieldPos[idiff<0]=0
            fieldNeg[idiff>0]=0
            
            fieldNeg=abs(fieldNeg)
            
            pwNeg=ipwc+fieldNeg
            pwPos=ipwc+fieldPos
            
            fieldComp=np.stack((ipwc,pwPos,pwNeg),axis=2)
                      
            
            cv2.imwrite(path+'diffPW/'+filename,np.around(fieldComp)) 
    
            
def checkData(root):
    """
    Test function which prints the filename of images with NaN values

    """
    setList=['train', 'validation', 'test']

    for s in setList:
        path=root+s+'/'
        for filename in os.listdir(path+'pwc/'):
            diff=cv2.imread(path+'diff/'+filename,1)
            for i in range(0,2):
                ide=diff[:,:,i]
                idea=np.isnan(np.sum(ide))
                if idea:
                    print('Diff '+i+' '+filename)
            for i in ['pwc','def']:
                idef=cv2.imread(path+i+'/'+filename,0)
                ide=np.sum(idef)
                idef_isnan=np.isnan(ide)
                if idef_isnan:
                    print(i+' '+filename)
                    
                    
                    
