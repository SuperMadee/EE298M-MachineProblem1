#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*
# Course: EE 298M - Foundations of Machine Learning
# Title: Machine Problem 1 - Removing Projective Distortion on Images

# Name: Ma. Madecheeen S. Pangaliman
# Student Number: 202220799
#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*


#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*
#Importing the needed libraries
#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*

# To run the image into another window, the program must depend on a backend

## I need to import matplotlib to use a renderer and a canvas
import matplotlib

## I use an Anti-Grain Geometry (Agg) C++ library as a renderer that requires
## a Tk canvas
matplotlib.use("TkAgg")

# This library contains the "argv[]" function that I needed to make sure that 
# my first command line argument will be passed to my Python script.  
import sys

# I need to import the OS module to interact with my Operating System because
# I want to identify my current directory where my dataset rests
import os

# I need to import the pyplot module for the visualization part of my program
import matplotlib.pyplot as plt

# I need to import the Image module from the Pillow library for image manipulation.
# It was called PIL to make it backward compatible with an older module 
# called Python Imaging Library (PIL)
from PIL import Image

# I need to import Numpy library for the matrix manipulations
import numpy as np

# I need to import the SciPy library to manipulate the data and visualize the data
# especially in terms of using the linalg function
import scipy

# I would be needing the griddata method to interpolate the 2D grid essential
# in creating the new image
from scipy.interpolate import griddata

#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*
#Defining the functions
#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*

# This function is responsible for translating the image into an array
# It uses a try-except block for exception handling (input part)
def load_image(im):
    
    try: 
        #From the PIL library, the Image.open method was used to read the
        #input image from the console
        imx = Image.open(im)
        #the function will return the equivalent array of the image
        return np.array(imx)
    #IOError - raised when the input/output operation fails
    except IOError:
        #if the program encounters an error in the input, it will return an
        #empty list
        return []
 
# This function is responsible for selecting the four points in the chosen image
# Please be reminded that only 4 points are needed to be selected, otherwise,
# there will be an error in the program in a form of a spinning wheel
# Note: The selection of the 4 points must be in order (CW or CCW)  
def select_four_points(im): 
     # The image selected will be shown      
     plt.imshow(im)
     # The xx variable will store the coordinated of the 4 points selected     
     xx = plt.ginput(4)
     # the coordinates of the points will be appended on the xx list
     xx.append(xx[0])
     # the function will return the list of coordinates of the selected points
     return xx
     
# This function is responsible for the generation of the ground truth coordinates 
# where x (as zzd) is the input and x'(as zz) is the output needed for the DLT  

def generate_GT(xx):
    # this block shows the figure created (red lines) based on the 
    # 4 points selected
    zzd = np.zeros((5,3))   
    for ii in range(len(xx)-1):         
        x1 =xx[ii][0]; y1=xx[ii][1]
        zzd[ii,0] = x1; zzd[ii,1] = y1; zzd[ii,2] = 1; 
        plt.plot([xx[ii][0],xx[ii+1][0]], [xx[ii][1],xx[ii+1][1]], 'ro-') 
    
    # this block shows the adjustments made (in a form of a green rectangle)
    # based on the selected 4 points to show correspondence from the zzd input
    # (projected) and zz output (straightened) 
    jj = 0
    aa = [0,0,1,0,1,3,0,3]
    zz = np.zeros((5,3))     
    for ii in range(len(zzd)-1):
            zz[ii,0] = zzd[aa[jj],0] 
            zz[ii,1] = zzd[aa[jj+1],1] 
            zz[ii,2] = 1;   
            jj = jj+2
    zz[4,:] = zz[0,:]
    for ii in range(4):      
        plt.plot([zz[ii,0],zz[ii+1,0]], [zz[ii,1],zz[ii+1,1]], 'go-')
    plt.show()
    return zz[0:4,:],zzd[0:4,:]    

# This function shows how the points are normalized in the program to 
# ensure that each input parameter (pixel) has a similar data distribution.
def normalize_points(zz):
    
    # create the necessary matrix needed to obtain the center of the region
    uu = zz.T
    ff_xx = np.ones(uu.shape)
    indices, = np.where(abs(uu[2,:]) > 10**-12)
    ff_xx[0:2,indices] = uu[0:2,indices]/uu[2,indices]
    ff_xx[2,indices]  = 1.
    
    # In normalization, you need to create the center of the region by taking
    # the mean of the points
    mu = np.mean(ff_xx[0:2,:],axis = 1)
    
    # Extending the obtained value to a vector
    mu_r = np.zeros((mu.shape[0],ff_xx.shape[1]))   
    for ii in range(ff_xx.shape[1]):
        mu_r[:,ii] = mu
    
    # Compute for the mean of the Euclidean distance of each point to the
    # center of the region 
    mu_dist = np.mean((np.sum((ff_xx[0:2] - mu_r)**2,axis =0))**0.5)

    # Obtain the scaling matrix
    scale =  (2**0.5/mu_dist)
    s0 = -scale*mu[0]
    s1 = -scale*mu[1]
    S = np.array([[scale, 0, s0],[0, scale, s1], [0, 0, 1]])
    
    # The normalized matrix is found by obtaining the dot product between the
    # scaling matrix and the points
    normalized_zz = S@ff_xx
    return normalized_zz, S
    
def compute_A(uu,vv):
    ## Note that uu = x' (ground truth) and vv = x (distorted)
    A = np.zeros((2*(uu.shape[0]+1),9))
    jj = 0
   
    for ii in range(uu.shape[0]+1):
        a = (np.zeros((1,3))[0] )     # added zeroes  
        b = (-uu[2,ii] * vv[:,ii])    # computing the first coefficient  
        c =  uu[1,ii] * vv[:,ii]      # computing the second coefficient
        d =  uu[2,ii] * vv[:,ii]      # computing the third coefficient
        f =  (-uu[0,ii]*vv[:,ii])     # computing the fourth coefficient
        
        # concatenate the obtained coefficients to build the matrix A
        # from A1 to A4
        row1 = np.concatenate((a, b, c), axis=None) 
        row2 = np.concatenate((d,a,f), axis=None)
        A[jj,:] = row1
        A[jj+1,:] = row2
        jj = jj+2
    return A

# This function shows how to compute for the homography (hh) given by the 
# matrices A, normalized points in original image (T1) and normalized points
# in translated image (T2)
def compute_homography(A,T1,T2):
    
    # Compute the null space of matrix A
    null_space_of_A = -scipy.linalg.null_space(A)
    
    # Reshaping the null space into 3x3 matrix
    hh_normalized = np.reshape(null_space_of_A,(3,3)) 
    
    # Obtain the value of the homography (hh) by using the dot product
    # between the inverse of T2 and the dot product of the reshaped null space
    # and T1
    hh = np.dot(np.linalg.inv(T2),np.dot(hh_normalized,T1))
    return hh

# The series of functions shows how the image is transformed using the 
# homography (hh) matrix obtained

# This function shows how to specify the bounds on the new image
# (mapping the bounds of the original image to the new image)
# See documentation for the assigned coordinates of each bound
def image_rebound(mm,nn,hh):
    W = np.array([[1, nn, nn, 1 ],[1, 1, mm, mm],[ 1, 1, 1, 1]])
    ws = np.dot(hh,W)
    xx = np.vstack((ws[2,:],ws[2,:],ws[2,:]))
    wsX =  np.round(ws/xx)
    bounds = [np.min(wsX[1,:]), np.max(wsX[1,:]),np.min(wsX[0,:]), np.max(wsX[0,:])]
    return bounds

# This function shows how to use DLT in transforming the image
def make_transform(imm,hh):   
    mm,nn = imm.shape[0],imm.shape[0]
    bounds = image_rebound(mm,nn,hh)
    nrows = bounds[1] - bounds[0]
    ncols = bounds[3] - bounds[2]
    s = max(nn,mm)/max(nrows,ncols)
    scale = np.array([[s, 0, 0],[0, s, 0], [0, 0, 1]])
    trasf = scale@hh
    trasf_prec =  np.linalg.inv(trasf)
    bounds = image_rebound(mm,nn,trasf)
    nrows = (bounds[1] - bounds[0]).astype(int)
    ncols = (bounds[3] - bounds[2]).astype(int)
    return bounds, nrows, ncols, trasf, trasf_prec

# This function shows how to remap the original image to a new image
# using sampling
def get_new_image(nrows,ncols,imm,bounds,trasf_prec,nsamples):
    
    # Starts with reshaping the original image into the new image
    xx  = np.linspace(1, ncols, ncols)
    yy  = np.linspace(1, nrows, nrows)
    [xi,yi] = np.meshgrid(xx,yy) 
    a0 = np.reshape(xi, -1,order ='F')+bounds[2]
    a1 = np.reshape(yi,-1, order ='F')+bounds[0]
    a2 = np.ones((ncols*nrows))
    uv = np.vstack((a0.T,a1.T,a2.T)) 
    new_trasf = np.dot(trasf_prec,uv)
    
    # Do some normalization 
    val_normalization = np.vstack((new_trasf[2,:],new_trasf[2,:],new_trasf[2,:]))
    newT = new_trasf/val_normalization
    
    # Sample the points from the original image
    xi = np.reshape(newT[0,:],(nrows,ncols),order ='F') 
    yi = np.reshape(newT[1,:],(nrows,ncols),order ='F')
    cols = imm.shape[1]
    rows = imm.shape[0]
    xxq  = np.linspace(1, rows, rows).astype(int)
    yyq  = np.linspace(1, cols, cols).astype(int)
    [x,y] = np.meshgrid(yyq,xxq)
    
    # Offset x and y relative to region origin.
    x = (x - 1).astype(int) 
    y = (y - 1).astype(int) 
        
    ix = np.random.randint(im.shape[1], size=nsamples)
    iy = np.random.randint(im.shape[0], size=nsamples)
    samples = im[iy,ix]
    
    # Interpolate the points of the original image (iy, ix) via the obtained
    # samples into the coordinates of the new image (yi, xi)
    int_im = griddata((iy,ix), samples, (yi,xi))
    
    #Plotting the new image as subplot
    fig = plt.figure(figsize=(9, 9))
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im)
    fig.add_subplot(rows, columns, 2) 
    plt.imshow(int_im.astype(np.uint8))
    plt.show()
    

#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*
#Main Program
#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*    

if __name__ == "__main__":
    
    # Upon running the program in the Anaconda prompt, the image number should
    # be declared after the filename: "python projective_distortion_MSP.py imnum
    # where imnum are image numbers (0,1,2,3...)
    # Note that image numbers depend on the images in the dataset arranged
    # in alphabetical order (ex: Image1 - 1, Image2 - 2)
    # The method sys.argv[1] denotes that the first line will be passed to the
    # script
    imnum= sys.argv[1]
    
    # The current working directory wll be stored in the 'wd' variable.
    wd = os.getcwd()
    
    # This line returns a list containing the names of the entries in the 
    # current working directory in arbitrary order
    ddir = os.listdir(wd)
    
    # For organization purposes, the images are stored in a folder (found in the
    # current directory) called 'dataset'. 
    # The program will verify if this folder exist.
    if 'dataset' not in ddir:
        
        # If the 'dataset' folder is not found, a display message will appear
        print(('oops there is no dataset folder') )
    else:  
        
        # If the 'dataset' folder is found, the program will get the filepath
        # of the image. 
        # For the dataset path, the word 'dataset' will be added to the 
        # contents of the current working directory
        dataset_path = os.path.join(wd,'dataset')
        
        # The images available will be the list of the images found in the
        # dataset folder.
        images_available=os.listdir(dataset_path)
        
        # Numbers will be assigned to the images located in the dataset folder
        image_building = images_available[int(float(imnum))]
        
        # The file path of the chosen image will be stored to the imagetoload
        # variable
        imagetoload = os.path.join(wd,'dataset',image_building)
        
        # Load the chosen image
        im =  load_image(imagetoload)
        
        # The 4 points (or corners) selected from the image will be stored in
        # the 'xx' variable. You may chose to see these coordinates by typing
        # xx in the console
        xx = select_four_points(im)
        
        # Generating the ground truth (zz and zzd) from the coordinates of the
        # 4 points
        zz, zzd = generate_GT(xx)
            
        ## Normalize points and return the scaling matrix
        # Normalize the points in the original image
        norm_points_distorted, T1_norm = normalize_points(zzd)
        # Normalize the points in the transformed image
        norm_points_GT, T2_norm= normalize_points(zz)
        
        # Computing the matrix A (composed of stacked A1, A2, A3 and A4)
        A = compute_A(norm_points_GT,norm_points_distorted)
        
        # Computing the homography 
        hh =  compute_homography(A,T1_norm,T2_norm)
        
        # Determining the needed information for the remapping of the original
        # image to the new image
        bounds, nrows, ncols,  trasf, trasf_prec = make_transform(im,hh)     
        nn,mm  = im.shape[0],im.shape[0]
        
        # The number of samples were considered to determine the spatial
        # resolution of the new image. Note that the number of samples we
        # choose from the image is dependent on the size of the image.
        if max(nn,mm)>1000:
            kk = 6
        else: 
            kk = 5
        
        # The exponent needed for determining the number of samples is only
        # limited to 5 or 6 because the size of the image is large
        nsamples = 10**kk      
        
        get_new_image(nrows,ncols,im,bounds,trasf_prec,nsamples)