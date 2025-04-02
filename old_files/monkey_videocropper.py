#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:48:59 2022

@author: amendez

NOTE: use conda environment "yolox"

"""


import os
import shutil
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import progressbar
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import VideoGear

import datetime
import pandas as pd

#% FUNCTION DEFINITIONS

#COUNT IN YOLO FILES STARTS AT 1, not 0!
def get_detected_list(txtdirpath):
    detectedlist=os.listdir(txtdirpath)
    detectedlist=[ int(a.split('_')[-1].split('.')[0]) for a in detectedlist]
    detectedlist.sort()
    return detectedlist


def fix_detected_filenames_stride(txtdirpath, stride, videofile):
    '''
    Yolo saves txt files with indices corresponding to the frame number, but if a stride was used the indexing is wrong. 
    The correct index is calculated by real_frame_idx=stride*(frame_idx-1)+1, remember that it starts counting on 1, not 0
    '''
    txtdirpathfixed=txtdirpath+'_fixed' #directory to save fixed txt yolo files
    if not os.path.exists(txtdirpathfixed):
        os.makedirs(txtdirpathfixed) 
    detectedlist=os.listdir(txtdirpath)
    detectedlist.sort()
    videoname=os.path.splitext(os.path.basename(videofile))[0]
    print('\n Fixing the frame indices with correct stride ===== \n')
    if stride>1:
        for file in progressbar.progressbar(detectedlist):
            n=int(file.split('_')[-1].split('.')[0]) #the frame number in the file, needs to be corrected according to the stride
            real_n=stride*(n-1)+1
            #os.rename(os.path.join(txtdirpath,file),os.path.join(txtdirpath,videoname+'_'+str(real_n)+'.txt'))
            shutil.copy2(os.path.join(txtdirpath,file), os.path.join(txtdirpathfixed,videoname+'_'+str(real_n)+'.txt'))
    return txtdirpathfixed        


def get_XY_coords_from_roi(rois):
    '''
    returns upper left and lower right corner of the superbox containing both detected boxes,in relative coords, 
    also the geometric center of such box 
    rois : numpy array in the order given by Yolov5
        mxn
        m=rows
        n=X,Y,Width,Height (relative numbers [0,1], as given by YOLOv5)
    '''
    if not isinstance(rois,np.ndarray):
        raise TypeError('the roi needs to be a numpy array')

    x1,x2,y1,y2=([] for i in range(4))
    for i in range(rois.shape[0]):
        x1.append(rois[i,0]-(rois[i,2]/2))
        y1.append(rois[i,1]-(rois[i,3]/2))
        x2.append(rois[i,0]+(rois[i,2]/2))
        y2.append(rois[i,1]+(rois[i,3]/2))    
    x1=np.min(x1)
    y1=np.min(y1)
    x2=np.max(x2)
    y2=np.max(y2)
    center=np.mean(rois,axis=0)[0:2]
    return x1,y1,x2,y2,center
    

# indexing starting at 1 since its the Yolo concc
def get_crop_coord_list(txtdirpath,videofile,numberofframes):
    #
    #returns an np array with coords: x1,x2,y1,y2,center0,center1
    #x1,x2,y1,y2 refer to the megabox corners containing the detected macaques
    # the coords are expressed in relative coords
    #
    detectedlist=get_detected_list(txtdirpath)
    videofile=os.path.basename(videofile)
    coords=np.zeros((numberofframes,6))
    x1,y1,x2,y2=(0 for i in range(4)) #initialize 
    center=[0,0]
    print('\n \n Getting the coordinate list of the cropped frames')
    for f in progressbar.progressbar(range(1,numberofframes+1)): #start counting on 1
        #if f%1000==0:    
        #    print(f'frame {f}')
        if f in detectedlist:
            rois=np.loadtxt(os.path.join(txtdirpath,os.path.splitext(videofile)[0]+'_'+str(f)+'.txt')) #load Yolov5 txt
            if rois.size==0:
                continue
            if len(rois.shape)==2: #if two monkeys were detected
                rois=np.delete(rois,0,axis=1) #delete the first column, class
            else:
                #print(f)
                rois=np.delete(rois,0)
                rois=rois.reshape(1,rois.shape[0])
            x1,y1,x2,y2,center=get_XY_coords_from_roi(rois)
            coords[f-1,:]=x1,y1,x2,y2,center[0],center[1] #start filling the coords array (f-1 because the frames start on 1)
        else: #no monkey detected in the frame, ergo use the previous' frame center                  
            #print(coords.shape)
            #print(x1,y1,x2,y2)
            coords[f-1,:]=x1,y1,x2,y2,center[0],center[1] #start filling the coords array
    return coords


    



#DEPRECATED FUNCTION
def crop_frame_OLD(frame,cropwidth,coordsvec):
    '''
    Parameters
    ----------
    frame : TYPE
        DESCRIPTION.
    cropwidth : TYPE
        DESCRIPTION.
    coordsvec : TYPE
        x1,y1,x2,y2,center[0],center[1] of the single metabox containing detected boxes, in relative coords [0,1]

    Returns
    -------
    None.
    '''
    pad=0 #initialize padding flag
    if max(frame.shape)<cropwidth:
        cropwidth=max(frame.shape)-10  
    #Coordinates of the center (now in real pixels)
    x=int(np.round(coordsvec[4]*frame.shape[1]))
    y=int(np.round(coordsvec[5]*frame.shape[0]))
    
    #do the megabox (containing both monkeys) fit inside the square dimensions defined by cropwidth?
    if cropwidth>((coordsvec[2]-coordsvec[0])*frame.shape[1]) and cropwidth>((coordsvec[3]-coordsvec[1])*frame.shape[0]):
        #OK, both monkey boxes fit inside the cropwidth square, so we can use the central coordinates    
        #frame=frame[(y-int(cropwidth/2)):(y+int(cropwidth/2)),(x-int(cropwidth/2)):(x+int(cropwidth/2)),:]
        x1=x-int(cropwidth/2)
        y1=y-int(cropwidth/2)
        x2=x+int(cropwidth/2)
        y2=y+int(cropwidth/2)           
    else: #the cropping is not enough to cover both monkeys, use the megabox
        #get the pixel coordinates of the megabox        
        x1=int(np.round(coordsvec[0]*frame.shape[1]))
        y1=int(np.round(coordsvec[1]*frame.shape[0]))
        x2=int(np.round(coordsvec[2]*frame.shape[1]))
        y2=int(np.round(coordsvec[3]*frame.shape[0]))
        alpha=max(x2-x1,y2-y1)#define the larger side and use that to get a square
        #if alpha is larger than the smaller dimension of the frame we need to pad it, 
        # since it wont fit inside the frame anyway         
        if alpha>min(frame.shape[0:2]): 
            pad=1 #padding flag, the actual padding will be applied later 
        else: #get a square
            x2=x1+alpha
            y2=y1+alpha
    #if the box is out of the boundaries of the original frame
    if any(np.array([x1,x2])<0):
        alpha=abs(min(x1,x2))
        x1=x1+alpha
        x2=x2+alpha
    if any(np.array([y1,y2])<0):
        alpha=abs(min(y1,y2))
        y1=y1+alpha
        y2=y2+alpha
    if any(np.array([x1,x2])>frame.shape[1]): #if any x is out of positive bounds
        alpha=max(x1-frame.shape[1],x2-frame.shape[1])
        x1=x1-alpha
        x2=x2-alpha
    if any(np.array([y1,y2])>frame.shape[0]): #if any x is out of positive bounds
        alpha=max(y1-frame.shape[0],y2-frame.shape[0])
        y1=y1-alpha
        y2=y2-alpha
    frame=frame[y1:y2,x1:x2,:]
    if pad==1:
       frame=pad_img_to_square(frame) 
   #print(frame.shape)
    frame=cv2.resize(frame,(cropwidth,cropwidth))
    return frame



def crop_frame(frame,coordsvec):
    '''
    THIS FUNCTION ONLY APPLIES TO TALL FRAMES SUCH AS THE VERTICAL ONES RECORDED IN THE WIRELESS PROJECT 2022
    
    Parameters
    ----------
    frame : TYPE
        DESCRIPTION.
    cropwidth : TYPE
        DESCRIPTION.
    coordsvec : TYPE
        x1,y1,x2,y2,center[0](X),center[1](Y) of the single metabox containing detected boxes, in relative coords [0,1]

    Returns
    -------
    None.
    '''
    pad=0 #initialize padding flag
    H=frame.shape[0]
    W=frame.shape[1]

    #Coordinates of the center (now in real pixels)
    fx1=int(np.round(coordsvec[0]*W))
    fx2=int(np.round(coordsvec[2]*W))
    fy1=int(np.round(coordsvec[1]*H))
    fy2=int(np.round(coordsvec[3]*H))    
    #fx=int(np.round(coordsvec[4]*W)) #CENTER x
    fy=int(np.round(coordsvec[5]*H)) #CENTER y
    
    #check the limits in order to use just the center as a reference
    if fy>=W/2 and fy<= H-(W/2) and (fy2-fy1)<=W: #if the limits are in the center area, so the cutting is straightforward
        x1=0
        x2=W
        y1=fy-int(W/2)
        y2=fy+int(W/2)        
    elif (fy2-fy1)>W:#if the height of the box is larger than the width 
        pad=1 #padding flag        
        x1=0
        x2=W
        y1=fy1
        y2=fy2              
    elif fy<H/2: #if the center falls in the upper part of the frame
        x1=0
        y1=0
        x2=W
        y2=W    
    elif fy>H-(W/2): #if the center falls in the lower part of the frame
        x1=0
        x2=W
        y1=H-W
        y2=H
    else:
        raise ValueError(f'error with coordinate vector: {coordsvec}')  
    frame=frame[y1:y2,x1:x2,:]
    if pad==1:
        #frame=frame[y1:y2,x1:x2,:]
        frame=pad_img_to_square(frame) 
        frame=cv2.resize(frame,(W,W))
    assert frame.shape[0]==W and frame.shape[1]==W    
    return frame
    
    


def pad_img_to_square(img):
    h,w,_=img.shape
    maxside=max(h,w)
    imgsq=np.zeros((maxside,maxside,3),'uint8')
    imgsq[:h,:w,:]=img
    return imgsq



def convert_from_ms( milliseconds ): 
	seconds, milliseconds = divmod(milliseconds,1000) 
	minutes, seconds = divmod(seconds, 60) 
	hours, minutes = divmod(minutes, 60) 
	days, hours = divmod(hours, 24) 
	seconds = seconds + milliseconds/1000 
	return int(days), int(hours), int(minutes), seconds 


def convert_hhmmss_to_seconds(time_str):
    '''
    time_str should be in the form "hh:mm:ss.sss"
    '''
    hours, minutes, seconds=time_str.split(':')
    total_seconds=(int(hours)*3600)+(int(minutes)*60)+float(seconds)
    return round(total_seconds,3)


def convert_datetime_format_to_seconds(datetimeobject):
    seconds=datetime.timedelta(hours=datetimeobject.hour, minutes=datetimeobject.minute,seconds=datetimeobject.second, 
                                   milliseconds=datetimeobject.microsecond/1000).total_seconds()
    return seconds


def ffmpeg_video_transformation(video_original, rootdir='/data/socialeyes/EthoProcessedVideos', overwrite=False):
    '''
    basedir=rootdir/$VIDEO/$CAMERAX
    '''
    if 'Camera0' in video_original:
        camera='Camera0'
    elif 'Camera1' in video_original:
        camera='Camera1'
    else:    
        camera='other'
    basedir=os.path.join(rootdir,os.path.basename(os.path.dirname(video_original)),camera)
    if not os.path.exists(basedir):
        os.makedirs(basedir) 
    videosmall=os.path.join(basedir,os.path.splitext(os.path.basename(video_original))[0]+'_small.mp4') #name of the output video
    if os.path.exists(videosmall) and overwrite==False:
        pass
    else:
        ffmpegstring = f'ffmpeg -i "{video_original}" -vf "fps=60, scale=iw/2:ih/2, eq=saturation=1.1,unsharp" -c:v mpeg4 -qscale:v 2 -vsync 1 "{videosmall}"'
        subprocess.call(ffmpegstring, shell=True)
    return basedir, videosmall


def macaquedetector3000(detect_script,weights,stride,basedir,source_video):
    detectstring=f'python {detect_script} --weights {weights} --source {source_video} --device=0 --line-thickness=3 \
        --iou-thres=0.5 --max-det=2 --save-txt --project {basedir} --vid-stride={stride}'
    subprocess.run(detectstring,shell=True)



def crop_transformed_video(basedir,source_video,stride,unsharp=False,timestamp=True,stabilize=False,suffix='_CROP',
                           filter_type='gaussian', filter_windoworsigma=20, overwrite=False):
    '''
    
    CAREFUL!!!  stabilization does not guarantees the correct timing !!!!! dont use it for time sensitive videos
    
    Remember that this requires the use of monkeydetector3000 to get the monkey coords
    
    basedir is basically where the exp dirs are, normally also the video to be cropped (video_small)
    
    filter_type either: None, "median", "gaussian"
    
    '''
    #create output video name
    if unsharp:
        outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'_unsharp.mp4')
    else:
        outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    
    #if out file already exists and overwrite=False then end function and return outvideofile name
    if os.path.exists(outvideofile) & (overwrite==False):
        return outvideofile
    
    
    #get the directory where the yolo labels were saved, might be a better way to do it
    exp=[a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir,a))]
    exp.sort()
    exp=exp[-1]
    txtdirpath=os.path.join(basedir,exp,'labels')

    if stride==1: #there is no need to correct the YOLO files when stride==1
        txtdirpathfixed=txtdirpath
    else:
    #FIX the coordinates indices (according to the stride used in the monkeydetector), it takes a while
        txtdirpathfixed=txtdirpath+'_fixed'
        if not os.path.exists(txtdirpathfixed):
            txtdirpathfixed=fix_detected_filenames_stride(txtdirpath, stride, source_video)

    #---------------------------------- load the video to be cropped with cv2
    cap = cv2.VideoCapture(source_video)
    #----------------------------------
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    #get the numpy array holding the coordinates "xl,yl,xr,yr,center[0],center[1]"
    coords=get_crop_coord_list(txtdirpathfixed,source_video,numberofframes)
    if filter_type=='median':
        coords=rolling_median_by_column(coords,window_size=filter_windoworsigma, output_equal_length=True)
    elif filter_type=='gaussian':
        coords=gaussian_filter1d(coords,sigma=filter_windoworsigma,axis=0)
    elif filter_type=='median':
        coords=uniform_filter1d(coords,size=filter_windoworsigma,axis=0)
    
    #create output video name
    if unsharp:
        outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'_unsharp.mp4')
    else:
        outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    
    #initialie CV2 videowriter
    out = cv2.VideoWriter(outvideofile,cv2.VideoWriter_fourcc(*'mp4v'), 60, (width,width))
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) #in case you want to apply it to each frame
    
    # initiate stabilizer object with default parameters
    stab = Stabilizer(smoothing_radius=40,border_type='reflect_101')

    #parameters to print the HHMMSSSS into the frame
    font = cv2.FONT_HERSHEY_COMPLEX
    org = (10, width-10) #x,y
    color = (0, 0, 255) #red on BGR    
    fontScale = 1
    thickness = 2

    frame_idx=0
    print('\n = = = cropping the video = = =')
    bar=progressbar.ProgressBar(max_value=numberofframes, redirect_stdout=True) #this is the class ProgressBar not the function progressbar, check the documentation
    while(1):
        ret, frame = cap.read() #Read Frame
        if not ret:
            print('No frames grabbed!')
            break
        
        if stabilize==True: #stabilize frames
            # send current frame to stabilizer for processing
            stabilized_frame = stab.stabilize(frame)
            # wait for stabilizer which still be initializing
            if stabilized_frame is None:
                continue #look at stabilize documentation and examples
                #stabilized_frame=frame
            framecrop=crop_frame(stabilized_frame,coords[frame_idx,:]) 
        else:#use regular frame
            framecrop=crop_frame(frame,coords[frame_idx,:])  
        
        if unsharp: #applying a sharpening mask
            framecrop=cv2.filter2D(framecrop, -1, kernel) 
        if timestamp:#write the timestamp 
            _,hh,mm,ss,=convert_from_ms(cap.get(cv2.CAP_PROP_POS_MSEC))#get timestamp
            hhmmss=f'{hh:02d}:{mm:02d}:{ss:05.3f}' #format timestamp string
            _=cv2.putText(framecrop, hhmmss, org, font,fontScale, color, thickness, 
                          cv2.LINE_AA,bottomLeftOrigin=False)
        out.write(framecrop)
        bar.update(frame_idx)
        frame_idx+=1
    cap.release()
    out.release()
    return outvideofile #string with the output video name





def crop_transformed_video_stabilize(basedir,source_video,stride,suffix='_CROP',
                                     filter_type='gaussian', filter_windoworsigma=20,
                                     overwrite=False):
    '''
    
    CAREFUL!!!  stabilization does not guarantees the correct timing !!!!! dont use it for time sensitive videos
    
    basedir is basically where the exp dirs are, normally also the video to be cropped (video_small)
    '''
    
    #if out file already exists and overwrite=False then end function and return outvideofile name
    outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    if os.path.exists(outvideofile) & (overwrite==False):
        return outvideofile
    
    #get the directory where the yolo labels were saved, might be a better way to do it
    exp=[a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir,a))]
    exp.sort()
    exp=exp[-1]
    txtdirpath=os.path.join(basedir,exp,'labels')

    if stride==1: #there is no need to correct the YOLO files when stride==1
        txtdirpathfixed=txtdirpath
    else:
    #FIX the coordinates indices (according to the stride used in the monkeydetector), it takes a while
        txtdirpathfixed=txtdirpath+'_fixed'
        if not os.path.exists(txtdirpathfixed):
            txtdirpathfixed=fix_detected_filenames_stride(txtdirpath, stride, source_video)
    
    #---------------------------------- load the video to be cropped with VideoGear
    options = {'SMOOTHING_RADIUS': 40, 'BORDER_TYPE': 'black'}
    stream_stab = VideoGear(source=source_video, stabilize=True, **options).start()
    cap = cv2.VideoCapture(source_video)
    #----------------------------------
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    #get the numpy array holding the coordinates "xl,yl,xr,yr,center[0],center[1]"
    coords=get_crop_coord_list(txtdirpathfixed,source_video,numberofframes)

    if filter_type=='median':
        coords=rolling_median_by_column(coords,window_size=filter_windoworsigma, output_equal_length=True)
    elif filter_type=='gaussian':
        coords=gaussian_filter1d(coords,sigma=filter_windoworsigma,axis=0)
    elif filter_type=='median':
        coords=uniform_filter1d(coords,size=filter_windoworsigma,axis=0)    
        
    #create output video name
    #outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    
    #initialie CV2 videowriter
    out = cv2.VideoWriter(outvideofile,cv2.VideoWriter_fourcc(*'mp4v'), 60, (width,width))

    frame_idx=0
    print('\n = = = cropping the video = = =')
    bar=progressbar.ProgressBar(max_value=numberofframes, redirect_stdout=True) #this is the class ProgressBar not the function progressbar, check the documentation
    while(1):
        
        frame =stream_stab.read()#this is the frame that will be used
        ret, frameCV2 = cap.read() #this for creating the timestamp, in theory should be in sync with stream_stab
        if frame is None:
            break              
        framecrop=crop_frame(frame,coords[frame_idx,:])

        out.write(framecrop)
        bar.update(frame_idx)
        frame_idx+=1
    cap.release()
    out.release()
    stream_stab.stop()
    return outvideofile #string with the output video name

    

def cut_video_clip(video_in, video_out, time_start, time_end,overwrite=False):
    '''
    time_start and time_end need to be in ffmpeg format 'hh:mm:ss.sss'
    
    '''
    assert os.path.exists(video_in), 'Input Video file not found'
    if os.path.exists(video_out) and overwrite==False:
        pass
    else:
        ffmpegstring = f'ffmpeg -i "{video_in}" -ss {time_start} -to {time_end} -c:v copy -c:a copy {video_out}' 
        subprocess.call(ffmpegstring, shell=True)
    




def cut_video_clip_crop(video_in, video_out, time_start, time_end,crop_ffmpeg_string,overwrite=False):
    '''
    time_start and time_end need to be in ffmpeg format 'hh:mm:ss.sss'
    
    '''
    assert os.path.exists(video_in), f'Input Video file not found, {video_in}'
    if os.path.exists(video_out) and overwrite==False:
        pass
    else:
        #ffmpegstring = f'ffmpeg -i "{video_in}" -ss {time_start} -to {time_end} -filter:v "crop={crop_ffmpeg_string}" -c:v copy -c:a copy {video_out}' 
        ffmpegstring = f'ffmpeg -i "{video_in}" -ss {time_start} -to {time_end} -filter:v "crop={crop_ffmpeg_string}" {video_out}' 
        subprocess.call(ffmpegstring, shell=True)




def generate_video_dataset_file(rootdir, output_file, classdir={'chilling':0,'foraging':1,'grooming':2,'moving':3}):
    '''
    Generate the video annotation TXT file, in accordance with MMACTION2 library.
    The absolute paths might need to be changed to relative paths, make tests.
    https://mmaction2.readthedocs.io
    '''
    filelist=[]
    #classdir={'chilling':0,'foraging':1,'grooming':2,'moving':3}
    for root, dirs, files in os.walk(rootdir,topdown=False):
        for file in files:
            filelist.append(os.path.join(root,file)+' '+str(classdir[os.path.split(root)[1]]))
   #write the list to a txt file 
    with open(output_file, 'w') as fp:
        fp.write('\n'.join(filelist))   
    return 


def rolling_window(array, window_size, freq, axis=0):
    shape = array.shape[:axis] + (array.shape[axis] - window_size + 1, window_size) + array.shape[axis+1:]
    strides = array.strides[:axis] + (array.strides[axis],) + array.strides[axis:]
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return np.take(rolled, np.arange(0,shape[axis],freq), axis=axis)


def rolling_median_by_column(array,window_size, output_equal_length=True):
    """
    Calculates the median by column
    input: 2D numpy array, RxC
    output: numpy array 
    """
    if len(array.shape) != 2:
        raise AssertionError ('array must be a bidimensional array')
    arr_windowed=rolling_window(array,window_size,freq=1,axis=0)
    #arr_windowed should have dimension Rx(window_size)xC
    arr_med= np.median(arr_windowed,axis=1)
    if output_equal_length:
        delta=array.shape[0]-arr_med.shape[0] #rows to repeat at the end in order to match dimension
        arr_med=np.repeat(arr_med, [1]*(arr_med.shape[0]-1) +[1+delta], axis=0)
    return arr_med



def create_clip_cut_table(tstartsecs,tendsecs, basetime=10):
    '''
    tstartsecs and tendsecs expressed in seconds, just as the CSV exported from BORIS
    basetime = desired duration of video segments in seconds

    returns numpy array with two columns, 
    '''
    tstartdhms=convert_from_ms(tstartsecs*1000)
    tstartlist=[tstartdhms[1],tstartdhms[2],tstartdhms[3]] #to be coherent get rid of days and format it as a list, hh mm ssss
    #seconds_integer=str(tstartlist[2]).split('.')[0]
    #seconds_decimal=str(tstartlist[2]).split('.')[1]
    #seconds_decimal =seconds_decimal+'0' if len(seconds_decimal)==2 else seconds_decimal
    #tstart=f'{tstartlist[0]:02d}'+f'{tstartlist[1]:02d}'+f'{int(seconds_integer):02d}'+f'{int(seconds_decimal):02d}'

    tenddhms=convert_from_ms(tendsecs*1000)
    tendlist=[tenddhms[1],tenddhms[2],tenddhms[3]]
    #seconds_integer=str(tendlist[2]).split('.')[0]
    #seconds_decimal=str(tendlist[2]).split('.')[1]
    #seconds_decimal =seconds_decimal+'0' if len(seconds_decimal)==2 else seconds_decimal
    #tend=f'{tendlist[0]:02d}'+f'{tendlist[1]:02d}'+f'{int(seconds_integer):02d}'+f'{int(seconds_decimal):02d}'

    #original duration of the clip (the clip will need to be chopped into smaller pieces if it is too large)
    duration=datetime.timedelta(days=0, seconds=tendsecs-tstartsecs, 
                 microseconds=0, milliseconds=0, 
                 minutes=0,hours=0, weeks=0).total_seconds()

    cut_table=[] #initialize empty list 
    rest_to_cut=duration #FLOAT, converted to seconds
    cutstartlist=tstartlist #initialization before the loops
    cutstartdt=datetime.datetime.strptime(str(cutstartlist[0])+':'+str(cutstartlist[1])+':'+str(cutstartlist[2])[:6], '%H:%M:%S.%f')
    basetimedt=datetime.timedelta(seconds=basetime) 
    while(rest_to_cut/basetime>=2): #loop 
        cutenddt=cutstartdt+basetimedt #update cutting point       
        temp1=f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}'
        temp2=f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}'
        cut_table.append([temp1,temp2])
        #update variables to continue the loop
        rest_to_cut=rest_to_cut-basetime #float
        cutstartdt=cutenddt #use the ending point as the new start cutting point
    if rest_to_cut/basetime<1 or (2>rest_to_cut/basetime>1 and rest_to_cut%basetime<basetime/2): 
        #make cutenddt the actual end of the clip (tend)
        cutenddt=datetime.datetime.strptime(str(tendlist[0])+':'+str(tendlist[1])+':'+str(tendlist[2])[:6], '%H:%M:%S.%f') 
        temp1=f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}'
        temp2=f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}'
        cut_table.append([temp1,temp2])
    elif 2>rest_to_cut/basetime>1 and rest_to_cut%basetime>=0.5: #just divide the remaining time by two and voila
        last_piece=rest_to_cut
        for _ in range(2):
            cutenddt=cutstartdt+datetime.timedelta(seconds=last_piece/2)
            temp1=f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}'
            temp2=f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}'
            cut_table.append([temp1,temp2])
            #update variables to continue the loop
            cutstartdt=cutenddt #use the ending point as the new start cutting point
    return cut_table
        



def open_boris_csv(file,behaviors=['Feeding','Resting','Grooming'],basetime=10):     
    df=pd.read_csv(file, sep=',')
    dflist=[] #initialize list for holding data
    for behavior in behaviors:
        df2=df[(df['Behavior type']=='STATE') & (df['Behavior']==behavior) & (df['Duration (s)']>1)]
        duration=np.sum(df2['Duration (s)'])
        #print(f'behav:{behavior}, sum:{duration}')
        
        for rownumber in range(len(df2)): #loop over ROWS
            sourcestr=df2.iloc[rownumber]['Source']         
            video1path=sourcestr.split(';')[0].split('player #1: ')[1]
            video2path=sourcestr.split(';')[1].split('player #2: ')[1] if len(sourcestr.split(';'))==3 else None            
            #get cut_table with cutting times
            cut_table=create_clip_cut_table(df2.iloc[rownumber]['Start (s)'],  df2.iloc[rownumber]['Stop (s)'],basetime=basetime)
            for cuttimes in cut_table: #iterate over each cutting time pair           
                dflist.append([video1path,video2path,cuttimes[0], cuttimes[1],behavior])
     
    dfboris=pd.DataFrame(data=dflist, columns=['video1','video2','start','stop','behavior'])
    return dfboris
        


def boris_correct_video_path(badpath,basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',basefilesuffix='small.mp4'):
    '''
    corrects the path of videos saved by Boris, probably in different partitions and so on, check that it applies to your case before using!
    
    '''
    #string manipulations to get the base name of the video, example '2022.09.02_12h12_Camera0'
    badpath=os.path.basename(badpath)
    cameranumber=badpath.split('_Camera')[1][0] #first character after 'Camera'
    session=badpath.split('_Camera')[0]
    videobasename=session+'_Camera'+cameranumber
    goodpath=os.path.join(basepath,session,'Camera'+cameranumber,videobasename+basefilesuffix)
    return goodpath




def boris_file_extract_clips(borisfile, behaviors=['Feeding','Resting','Grooming'],basetime=10,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                basefilesuffix='_small.mp4'):
    
    #open file and separate the video lengths according to the basetime in order to create clips
    dfboris=open_boris_csv(borisfile,behaviors,basetime)
    #correct the video paths
    for rownumber in range(len(dfboris)): 
        dfboris.iloc[rownumber]['video1']=boris_correct_video_path(dfboris.iloc[rownumber]['video1'],basepath,basefilesuffix)
        dfboris.iloc[rownumber]['video2']=boris_correct_video_path(dfboris.iloc[rownumber]['video2'],basepath,basefilesuffix)

    for rownumber in range(len(dfboris)): 
        for videon in ['video1','video2']:
        #get videobasename = = = = = = = = = = = = = = = = = = = = =
            temppath=dfboris.iloc[rownumber][videon]
            temppath=os.path.basename(temppath)
            cameranumber=temppath.split('_Camera')[1][0] #first character after 'Camera'
            session=temppath.split('_Camera')[0] # the actual session name, important for path manipulation
            videobasename=session+'_Camera'+cameranumber
            
        #create video_out filename (str), for each clip = = = = = = =        
            temptimestr=dfboris.iloc[rownumber]['start'].split(':') #temporary 
            cutstartdt=datetime.datetime.strptime(dfboris.iloc[rownumber]['start'], '%H:%M:%S.%f')
            cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'

            temptimestr=dfboris.iloc[rownumber]['stop'].split(':') #temporary 
            cutstopdt=datetime.datetime.strptime(dfboris.iloc[rownumber]['stop'], '%H:%M:%S.%f')
            cutstopstr=f'{cutstopdt.hour:02d}'+f'{cutstopdt.minute:02d}'+f'{cutstopdt.second:02d}'+f'{int(cutstopdt.microsecond/1000):03d}'

            video_out=os.path.join(os.path.join(basepath,session,'Camera'+cameranumber),
                                   'clips',cutstartstr+'_'+ cutstopstr,
                                   videobasename+'_'+cutstartstr+'_'+ cutstopstr+'.mp4')
            
            #create directory tree if it doesnt exist
            if not os.path.exists(os.path.dirname(video_out)):
                os.makedirs(os.path.dirname(video_out))       
             
            #check if videosmall exists, otherwise create it with ffmpeg (takes a while)    
            videosmall= os.path.join(basepath,session,'Camera'+cameranumber,videobasename+basefilesuffix)
            if not os.path.exists(videosmall):
                #=========== CONVERT WHOLE VIDEO (not just the clips), dont overwrite ====================
                #DONT OVERWRITE if video is already there 
                video_in=os.path.join('/data/socialeyes/EtholoopData',session,videobasename+'.avi' )
                print(f'***** ==== procesing: {video_in} ==== ***** \n')
                basedir,videosmall=ffmpeg_video_transformation(video_in,
                                                               rootdir=basepath,
                                                               overwrite=False)                
 
        #cut video clips, following the cut timestamps in the dataframe, row by row    
            cut_video_clip(videosmall, video_out,
                           dfboris.iloc[rownumber]['start'],
                           dfboris.iloc[rownumber]['stop'],  #in "hh:mm:ss.sss" format
                           overwrite=False)       


            
            #=========== apply Macaquedetector3000 on the CUT clips =================================    
            stride=1
            if not os.path.exists(os.path.join(os.path.dirname(video_out),'exp')):
                macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                                    weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                                    stride=stride,
                                    basedir=os.path.dirname(video_out),
                                    source_video=video_out)        
                
            #=========== Crop clips according to the detected macaques location ====================                            
            croppedclip_stabilized=crop_transformed_video_stabilize(basedir=os.path.dirname(video_out),
                                 source_video=video_out,
                                 stride=stride,
                                 suffix='_stab_',
                                 filter_type='gaussian', 
                                 filter_windoworsigma=20,
                                 overwrite=False)     
            
            croppedclip=crop_transformed_video(basedir=os.path.dirname(video_out),
                                               source_video=video_out,
                                               stride=stride,
                                               unsharp=False,
                                               timestamp=False,
                                               stabilize=False,
                                               suffix='_',
                                               filter_type='gaussian', 
                                               filter_windoworsigma=20,
                                               overwrite=False)

            for clip in [croppedclip_stabilized,croppedclip]:
             #========== copy the cropped clip to the appropriate folder             
                destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/datasetXXX/dataset_all',
                                                      dfboris.iloc[rownumber]['behavior']) 
                        
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                
                cropfilefinalpath=os.path.join(destination_folder,os.path.basename(clip))
                if not os.path.exists(cropfilefinalpath):    
                    shutil.copy2(clip,cropfilefinalpath)         
                    
                
                
                
     
    '''
                #=========== apply Macaquedetector3000 on the CUT clips =================================    
                stride=1
                if not os.path.exists(os.path.join(os.path.dirname(video_out),'exp')):
                    macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                                        weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                                        stride=stride,
                                        basedir=os.path.dirname(video_out),
                                        source_video=video_out)        
                    
                 #=========== Crop clips according to the detected macaques location ====================                            
                    croppedclip=crop_transformed_video_stabilize(basedir=os.path.dirname(video_out),
                                          source_video=video_out,
                                          stride=stride,
                                          suffix='_s_',
                                          filter_type='gaussian', 
                                          filter_windoworsigma=20)     


                 #========== copy the cropped clip to the appropriate folder    
                    destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset3/dataset_all',
                                                          dfboris.iloc[rownumber]['behavior']) 
                            
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                        
                    shutil.copy2(croppedclip,os.path.join(destination_folder,os.path.basename(croppedclip)))
    '''                    
     
        
     
            
       
def boris_file_extract_clips_negativeclass(borisfile, behaviors=['Feeding','Resting','Grooming'],basetime=10,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                basefilesuffix='_small.mp4',
                                negativeclassdir='Other'):     
    df=pd.read_csv(borisfile, sep=',')
    dflist=[] #initialize list for holding data
    df2=df[(df['Behavior type']=='STATE') & (df['Behavior'].isin(behaviors))]
    start=[]
    stop=[]
    dflist=[]
    for rownumber in range(len(df2)-1):
        if (df2.iloc[rownumber+1]['Start (s)']-df2.iloc[rownumber]['Stop (s)'])>10:
                #start.append(np.ceil(df2.iloc[rownumber]['Stop (s)']))
                #stop.append(np.floor(df2.iloc[rownumber+1]['Start (s)']))
        
                sourcestr=df2.iloc[rownumber]['Source']         
                video1path=sourcestr.split(';')[0].split('player #1: ')[1]
                video2path=sourcestr.split(';')[1].split('player #2: ')[1] if len(sourcestr.split(';'))==3 else None            

                cut_table=create_clip_cut_table(np.ceil(df2.iloc[rownumber]['Stop (s)']),  
                                                np.floor(df2.iloc[rownumber+1]['Start (s)']), #! pay attention to the +1
                                                basetime=basetime)
                for cuttimes in cut_table: #iterate over each cutting time pair           
                    dflist.append([video1path,video2path,cuttimes[0], cuttimes[1]])
    dfboris=pd.DataFrame(data=dflist, columns=['video1','video2','start','stop'])


    for rownumber in range(len(dfboris)): 
        dfboris.iloc[rownumber]['video1']=boris_correct_video_path(dfboris.iloc[rownumber]['video1'],basepath,basefilesuffix)
        dfboris.iloc[rownumber]['video2']=boris_correct_video_path(dfboris.iloc[rownumber]['video2'],basepath,basefilesuffix)

    for rownumber in range(len(dfboris)): 
        for videon in ['video1','video2']:
        #get videobasename = = = = = = = = = = = = = = = = = = = = =
            temppath=dfboris.iloc[rownumber][videon]
            temppath=os.path.basename(temppath)
            cameranumber=temppath.split('_Camera')[1][0] #first character after 'Camera'
            session=temppath.split('_Camera')[0] # the actual session name, important for path manipulation
            videobasename=session+'_Camera'+cameranumber
            
        #create video_out filename (str), for each clip = = = = = = =        
            temptimestr=dfboris.iloc[rownumber]['start'].split(':') #temporary 
            cutstartdt=datetime.datetime.strptime(dfboris.iloc[rownumber]['start'], '%H:%M:%S.%f')
            cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'

            temptimestr=dfboris.iloc[rownumber]['stop'].split(':') #temporary 
            cutstopdt=datetime.datetime.strptime(dfboris.iloc[rownumber]['stop'], '%H:%M:%S.%f')
            cutstopstr=f'{cutstopdt.hour:02d}'+f'{cutstopdt.minute:02d}'+f'{cutstopdt.second:02d}'+f'{int(cutstopdt.microsecond/1000):03d}'

            video_out=os.path.join(os.path.join(basepath,session,'Camera'+cameranumber),
                                   'clips',cutstartstr+'_'+ cutstopstr,
                                   videobasename+'_'+cutstartstr+'_'+ cutstopstr+'.mp4')
            
            #create directory tree if it doesnt exist
            if not os.path.exists(os.path.dirname(video_out)):
                os.makedirs(os.path.dirname(video_out))       
             
            #check if videosmall exists, otherwise create it with ffmpeg (takes a while)    
            videosmall= os.path.join(basepath,session,'Camera'+cameranumber,videobasename+basefilesuffix)
            if not os.path.exists(videosmall):
                #=========== CONVERT WHOLE VIDEO (not just the clips), dont overwrite ====================
                #DONT OVERWRITE if video is already there 
                video_in=os.path.join('/data/socialeyes/EtholoopData',session,videobasename+'.avi' )
                print(f'***** ==== procesing: {video_in} ==== ***** \n')
                basedir,videosmall=ffmpeg_video_transformation(video_in,
                                                               rootdir=basepath,
                                                               overwrite=False)                
 
        #cut video clips, following the cut timestamps in the dataframe, row by row    
            cut_video_clip(videosmall, video_out,
                           dfboris.iloc[rownumber]['start'],
                           dfboris.iloc[rownumber]['stop'],  #in "hh:mm:ss.sss" format
                           overwrite=False)       



            #=========== apply Macaquedetector3000 on the CUT clips =================================    
            stride=1
            if not os.path.exists(os.path.join(os.path.dirname(video_out),'exp')):
                macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                                    weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                                    stride=stride,
                                    basedir=os.path.dirname(video_out),
                                    source_video=video_out)        
                
            #=========== Crop clips according to the detected macaques location ====================                            
            croppedclip=crop_transformed_video_stabilize(basedir=os.path.dirname(video_out),
                                 source_video=video_out,
                                 stride=stride,
                                 suffix='_s_',
                                 filter_type='gaussian', 
                                 filter_windoworsigma=20,
                                 overwrite=False)     


         #========== copy the cropped clip to the appropriate folder             
            destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset5/dataset_all',
                                                  negativeclassdir) 
                    
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            cropfilefinalpath=os.path.join(destination_folder,os.path.basename(croppedclip))
            if not os.path.exists(cropfilefinalpath):    
                shutil.copy2(croppedclip,cropfilefinalpath)
                
            
            '''
            #=========== apply Macaquedetector3000 on the CUT clips =================================    
            stride=1
            if not os.path.exists(os.path.join(os.path.dirname(video_out),'exp')):
                macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                                    weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                                    stride=stride,
                                    basedir=os.path.dirname(video_out),
                                    source_video=video_out)        
                
             #=========== Crop clips according to the detected macaques location ====================                            
                croppedclip=crop_transformed_video_stabilize(basedir=os.path.dirname(video_out),
                                      source_video=video_out,
                                      stride=stride,
                                      suffix='_s_',
                                      filter_type='gaussian', 
                                      filter_windoworsigma=20)     


             #========== copy the cropped clip to the appropriate folder    
                destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset3/dataset_all',
                                                      dfboris.iloc[rownumber]['behavior']) 
                        
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                    
                shutil.copy2(croppedclip,os.path.join(destination_folder,os.path.basename(croppedclip)))
            '''
            
            



def pad_video(basedir, source_video, suffix='_CROP', overwrite=False):
    '''
    
    CAREFUL!!!  stabilization does not guarantees the correct timing !!!!! dont use it for time sensitive videos
    
    basedir is basically where the exp dirs are, normally also the video to be cropped (video_small)
    '''
    
    #if out file already exists and overwrite=False then end function and return outvideofile name
    outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    if os.path.exists(outvideofile) & (overwrite==False):
        return outvideofile
    
    
    """
    #get the directory where the yolo labels were saved, might be a better way to do it
    exp=[a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir,a))]
    exp.sort()
    exp=exp[-1]
    txtdirpath=os.path.join(basedir,exp,'labels')

    if stride==1: #there is no need to correct the YOLO files when stride==1
        txtdirpathfixed=txtdirpath
    else:
    #FIX the coordinates indices (according to the stride used in the monkeydetector), it takes a while
        txtdirpathfixed=txtdirpath+'_fixed'
        if not os.path.exists(txtdirpathfixed):
            txtdirpathfixed=fix_detected_filenames_stride(txtdirpath, stride, source_video)
    
    #---------------------------------- load the video to be cropped with VideoGear
    options = {'SMOOTHING_RADIUS': 40, 'BORDER_TYPE': 'reflect_101'}
    stream_stab = VideoGear(source=source_video, stabilize=True, **options).start()
    cap = cv2.VideoCapture(source_video)
    #----------------------------------
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    #get the numpy array holding the coordinates "xl,yl,xr,yr,center[0],center[1]"
    coords=get_crop_coord_list(txtdirpathfixed,source_video,numberofframes)

    if filter_type=='median':
        coords=rolling_median_by_column(coords,window_size=filter_windoworsigma, output_equal_length=True)
    elif filter_type=='gaussian':
        coords=gaussian_filter1d(coords,sigma=filter_windoworsigma,axis=0)
    elif filter_type=='median':
        coords=uniform_filter1d(coords,size=filter_windoworsigma,axis=0)    
    """    
    #create output video name
    #outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(source_video))[0]+suffix+'.mp4')
    
    
    cap = cv2.VideoCapture(source_video)
    #----------------------------------
    numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    maxside=max(width,height)
    #initialie CV2 videowriter
    out = cv2.VideoWriter(outvideofile,cv2.VideoWriter_fourcc(*'mp4v'), 60, (maxside,maxside))

    frame_idx=0
    print('\n = = = padding video = = =')
    while(1):
        ret, frameCV2 = cap.read() #this for creating the timestamp, in theory should be in sync with stream_stab
        if not ret:
            break              

        framepad=pad_img_to_square(frameCV2)
        out.write(framepad)
        frame_idx+=1
    cap.release()
    out.release()
    return outvideofile #string with the output video name
              

def genericvideo_extract_clips(annotation_file, behaviors=['Feeding','Resting','Grooming','Other'],basetime=10,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                ffmpegconvert=False,
                                basefilesuffix='_small.mp4'):
    
    '''
    Parameters
    ----------
    annotation_file : 
        DESCRIPTION.
        the annotation file needs to be an excel file with the columns 
       ' video	macrocategory	modifier	time_in	time_out'

    behaviors : TYPE, optional
        DESCRIPTION. The default is ['Feeding','Resting','Grooming'].
    basetime : TYPE, optional
        DESCRIPTION. The default is 10.
    basepath : TYPE, optional
        DESCRIPTION. 
        
        The default is '/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP'.
    ffmpegconvert : TYPE, optional
        DESCRIPTION. The default is False.
    basefilesuffix : TYPE, optional
        DESCRIPTION. The default is '_small.mp4'.

    Returns
    -------
    None.

    '''
    
    df=pd.read_excel(annotation_file) 
     
    
    dflist=[]
    for behavior in behaviors:
        df2=df[(df['macrocategory']==behavior) ]
        
        for rownumber in range(len(df2)): #loop over ROWS      
            #get cut_table with cutting times
            start=convert_hhmmss_to_seconds(df2.iloc[rownumber]['time_in'])
            stop=convert_hhmmss_to_seconds(df2.iloc[rownumber]['time_out'])
            cut_table=create_clip_cut_table(start,stop,basetime=basetime)
            for cuttimes in cut_table: #iterate over each cutting time pair           
                dflist.append([df2.iloc[rownumber]['video'],cuttimes[0], cuttimes[1],behavior])
     
    dfcut=pd.DataFrame(data=dflist, columns=['video','start','stop','behavior'])    

    for rownumber in range(len(dfcut)):
        #get videobasename = = = = = = = = = = = = = = = = = = = = =
        videopath=dfcut.iloc[rownumber]['video']


        #create video_out filename (str), for each clip = = = = = = =        
        temptimestr=dfcut.iloc[rownumber]['start'].split(':') #temporary 
        cutstartdt=datetime.datetime.strptime(dfcut.iloc[rownumber]['start'], '%H:%M:%S.%f')
        cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'

        temptimestr=dfcut.iloc[rownumber]['stop'].split(':') #temporary 
        cutstopdt=datetime.datetime.strptime(dfcut.iloc[rownumber]['stop'], '%H:%M:%S.%f')
        cutstopstr=f'{cutstopdt.hour:02d}'+f'{cutstopdt.minute:02d}'+f'{cutstopdt.second:02d}'+f'{int(cutstopdt.microsecond/1000):03d}'
        
        videobasename=os.path.splitext(os.path.basename(dfcut.iloc[rownumber]['video']))[0]

        video_out=os.path.join(os.path.join(basepath,videobasename),
                               'clips',cutstartstr+'_'+ cutstopstr,
                               videobasename+'_'+cutstartstr+'_'+ cutstopstr+'.mp4')
            
        #create directory tree if it doesnt exist
        if not os.path.exists(os.path.dirname(video_out)):
            os.makedirs(os.path.dirname(video_out))       

        
        #cut video clips, following the cut timestamps in the dataframe, row by row    
        cut_video_clip(videopath, video_out,
                       dfcut.iloc[rownumber]['start'],
                       dfcut.iloc[rownumber]['stop'],  #in "hh:mm:ss.sss" format
                       overwrite=False)       

        #pad videos so that the output is square
        paddedvideo=pad_video(os.path.dirname(video_out), video_out, suffix='_PAD', overwrite=False)
    
        
        #========== copy the cropped clip to the appropriate folder             
        destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset5/dataset_all',
                                              dfcut.iloc[rownumber]['behavior']) 
                   
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        filefinalpath=os.path.join(destination_folder,os.path.basename(paddedvideo))
        if not os.path.exists(filefinalpath):    
            shutil.copy2(paddedvideo,filefinalpath)            
    
    
    




def ballesta_extract_clips(annotation_file, behaviors=['Feeding','Resting','Grooming', 'Moving','Other'], basetime=10,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP'):
    
    
    df=pd.read_excel(annotation_file)     
    
    dflist=[]
    for behavior in behaviors:
        df2=df[(df['macrocategory']==behavior)]
        
        for rownumber in range(len(df2)): #loop over ROWS      
            #get cut_table with cutting times
            start=convert_hhmmss_to_seconds(df2.iloc[rownumber]['time_in'])
            stop=convert_hhmmss_to_seconds(df2.iloc[rownumber]['time_out'])
            cut_table=create_clip_cut_table(start,stop,basetime=basetime)
            for cuttimes in cut_table: #iterate over each cutting time pair           
                dflist.append([df2.iloc[rownumber]['video'],cuttimes[0], cuttimes[1],behavior,df2.iloc[rownumber]['modifier']])
     
    dfcut=pd.DataFrame(data=dflist, columns=['video','start','stop','behavior','square'])    

    for rownumber in range(len(dfcut)):
        #get videobasename = = = = = = = = = = = = = = = = = = = = =
        videopath=dfcut.iloc[rownumber]['video']


        #create video_out filename (str), for each clip = = = = = = =        
        temptimestr=dfcut.iloc[rownumber]['start'].split(':') #temporary 
        cutstartdt=datetime.datetime.strptime(dfcut.iloc[rownumber]['start'], '%H:%M:%S.%f')
        cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'

        temptimestr=dfcut.iloc[rownumber]['stop'].split(':') #temporary 
        cutstopdt=datetime.datetime.strptime(dfcut.iloc[rownumber]['stop'], '%H:%M:%S.%f')
        cutstopstr=f'{cutstopdt.hour:02d}'+f'{cutstopdt.minute:02d}'+f'{cutstopdt.second:02d}'+f'{int(cutstopdt.microsecond/1000):03d}'
        
        videobasename=os.path.splitext(os.path.basename(dfcut.iloc[rownumber]['video']))[0]

        video_out=os.path.join(os.path.join(basepath,'ballesta',videobasename),
                               videobasename+'_'+'f'+str(dfcut.iloc[rownumber]['square'])+'_'+
                               cutstartstr+'_'+ cutstopstr+'.mp4')
            
        #create directory tree if it doesnt exist
        if not os.path.exists(os.path.dirname(video_out)):
            os.makedirs(os.path.dirname(video_out))       

        if dfcut.iloc[rownumber]['square']==1:
            print('============\n')
            print(video_out)
            print('============\n')
            #cut video clips, following the cut timestamps in the dataframe, row by row    
            cut_video_clip_crop(video_in=videopath, video_out=video_out,
                           time_start=dfcut.iloc[rownumber]['start'],
                           time_end=dfcut.iloc[rownumber]['stop'],  #in "hh:mm:ss.sss" format
                           crop_ffmpeg_string='990:990:297:90', # "crop=out_w:out_h:x:y" WINDOW 1 -------
                           overwrite=False)       
        elif dfcut.iloc[rownumber]['square']==2:
            cut_video_clip_crop(video_in=videopath, video_out=video_out,
                           time_start=dfcut.iloc[rownumber]['start'],
                           time_end=dfcut.iloc[rownumber]['stop'],  #in "hh:mm:ss.sss" format
                           crop_ffmpeg_string='660:660:1056:420', # "crop=out_w:out_h:x:y" WINDOW 2 --------
                           overwrite=False)   

        
        #========== copy the cropped clip to the appropriate folder             
        destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset6/dataset_all',
                                              dfcut.iloc[rownumber]['behavior']) 
                   
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        filefinalpath=os.path.join(destination_folder,os.path.basename(video_out))
        if not os.path.exists(filefinalpath):    
            shutil.copy2(video_out,filefinalpath)            
    
    








#%%  Conversion of Eloise's files, march 2024



def list_files_with_extension(folders, target_extension):
    result_files = []
    for folder in folders:
        print(f"Searching in folder: {folder}")
        for root, dirs, files in os.walk(folder):
            print(f"  Current directory: {root}")
            for file in files:
                if file.lower().endswith(target_extension.lower()):
                    file_path = os.path.join(root, file)
                    result_files.append(file_path)
                    print(f"    Found file: {file_path}")
    return result_files


    
input_folders = ['/data/socialeyes/ETHOCAGE/Vocalizations_Old/Vocalizations-JULY2023/Betta/Etholoop/2023.07.24_10h02',
'/data/socialeyes/ETHOCAGE/Vocalizations_Old/Vocalizations-JULY2023/Samo/Etholoop/2023.07.19_10h00',
'/data/socialeyes/ETHOCAGE/Vocalizations_Old/Vocalizations-JULY2023/Samo/Etholoop/2023.07.21_10h02']
target_extension = '.avi'

result_files = list_files_with_extension(input_folders, target_extension)




tic=time.time()
for video_original in result_files:


    basedir,videosmall=ffmpeg_video_transformation(video_original,
                                                   rootdir='/data/socialeyes/EthoProcessedVideos/Eloise_convertedvideos_march2024')
    
    stride=3
    macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                        weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                        stride=stride,
                        basedir=basedir,
                        source_video=videosmall)
    
    crop_transformed_video(basedir=basedir,
                           source_video=videosmall,
                           stride=stride,
                           unsharp=True)

toc=time.time()-tic
 


    
#%% September 2023

#open annotation file with time in hh:mm:ssss and save it as a BORIS style csv with the time in seconds 
file='/home/amendez/Documents/ML/action_recognition/video_annotations_moving_20220902_test.ods'
#df=pd.read_csv(file, sep=',')
df=pd.read_excel(file)    
df['Start (s)']=df['Start (s)'].apply(convert_datetime_format_to_seconds)
df['Stop (s)']=df['Stop (s)'].apply(convert_datetime_format_to_seconds)
df['Duration (s)']=df['Stop (s)']-df['Start (s)']
df.to_csv('/home/amendez/Documents/ML/action_recognition/video_annotations_moving_20220902_test_fixedtime.csv')


tic=time.time()
boris_file_extract_clips('/home/amendez/Documents/ML/action_recognition/video_annotations_moving_20220902_test_fixedtime.csv', 
                            behaviors=['Moving'],basetime=5,
                            basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                            basefilesuffix='_small.avi')
toc=time.time()-tic



dfboris=open_boris_csv('/home/amendez/Documents/ML/action_recognition/video_annotations_moving_20220902_test_fixedtime.csv',
                       behaviors=['Moving'],basetime=5)



#%% April 2023 
# Process annika naturalistic videos and create DATASET5 



#convert naturalistic videos to 60FPS (they are originally 30fps)
original_videos=['/data/AM_data/ActionRecognition/annika_naturalistic_videos/732008_173332.mpg',
'/data/AM_data/ActionRecognition/annika_naturalistic_videos/6232008_173647.mpg',
'/data/AM_data/ActionRecognition/annika_naturalistic_videos/6152008_123658.mpg',
'/data/AM_data/ActionRecognition/annika_naturalistic_videos/5212008_160351.mpg',
'/data/AM_data/ActionRecognition/annika_naturalistic_videos/792008_111617.mpg',
'/data/AM_data/ActionRecognition/annika_naturalistic_videos/712008_170839.mpg']



for input_video in original_videos:
    print(input_video)
    output_video=os.path.splitext(input_video)[0]+'_60fps.mp4'
    ffmpegstring = f'ffmpeg -i "{input_video}" -vf "fps=60, eq=saturation=1.1" -c:v mpeg4 -an -qscale:v 2 -vsync 1 "{output_video}"'
    subprocess.call(ffmpegstring, shell=True)



genericvideo_extract_clips('/home/amendez/Documents/ML/action_recognition/video_annotations_apr2023.ods', 
                           behaviors=['Feeding','Resting','Grooming','Other'],
                           basetime=5,
                           basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                           ffmpegconvert=False,
                           basefilesuffix='_small.mp4')
        
    
# split the data folder
import splitfolders
splitfolders.ratio('/data/AM_data/ActionRecognition/datasets/dataset5/dataset_all', 
                   output="/data/AM_data/ActionRecognition/datasets/dataset5/", 
                   seed=1337, ratio=(0.8, 0.1,0.1)) 



dataset='dataset5'
vec=['val','train','test']
for partition in vec:
    generate_video_dataset_file(os.path.join('/data/AM_data/ActionRecognition/datasets/',dataset,partition), 
                                os.path.join('/data/AM_data/ActionRecognition/datasets/',dataset, partition+'.txt'),
                                classdir={'Feeding':0,'Grooming':1,'Other':2,'Resting':3})


os.symlink('/data/AM_data/ActionRecognition/datasets/dataset5','/home/amendez/Documents/clonedrepos/mmaction2/data/dataset5')


#%% May 2023, convert BALLESTA videos to 60fps

tic=time.time()

ballesta_videos=[entry.path for entry in os.scandir('/data/AM_data/ActionRecognition/ballesta_videos/detections') if entry.is_file()]


outdir='/data/AM_data/ActionRecognition/ballesta_videos/detections_60fps'

for input_video in ballesta_videos:
    print(input_video)
    output_video=os.path.splitext(input_video)[0]+'_60fps.mp4'
    ffmpegstring = f'ffmpeg -i "{input_video}" -vf "fps=60" -c:v mpeg4 -an -qscale:v 2 -vsync 1 "{output_video}"'
    subprocess.call(ffmpegstring, shell=True)

toc=time.time()-tic

# Process ballesta videos into DATASET6

tic=time.time()
ballesta_extract_clips(annotation_file='/home/amendez/Documents/ML/action_recognition/video_annotations_june2023_ballesta.ods', 
                       behaviors=['Feeding','Resting','Grooming', 'Moving','Other'], 
                       basetime=5,
                       basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP')

toc=time.time()-tic


#%% jan 2023 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#borisfile='/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.02.Marine.for.csv'
#'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.02.Marine.for.csv']


tic=time.time()
borisfilelist=['/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.06_Eloise.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.07_Eloise.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.23_Giulia.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.08.25_Elsa.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.02.Marine.for.csv']



for borisfile in borisfilelist:
    boris_file_extract_clips(borisfile, 
                                behaviors=['Feeding','Resting','Grooming'],basetime=5,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                basefilesuffix='_small.mp4')
    
    
    boris_file_extract_clips_negativeclass(borisfile, 
                                           behaviors=['Feeding','Resting','Grooming'],
                                           basetime=5,
                                    basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                    basefilesuffix='_small.mp4')   
toc=time.time()-tic    
    



tic=time.time()
borisfilelist=['/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.08.25_Elsa.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.02.Marine.for.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.06_Eloise.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.07_Eloise.csv',
'/data/socialeyes/ETHOCAGE/PILOT_Betta_Samovar/Boris_projects/Exports_Andres/2022.09.23_Giulia.csv']



for borisfile in borisfilelist:
    boris_file_extract_clips(borisfile, 
                                behaviors=['Feeding','Resting','Grooming'],basetime=5,
                                basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                basefilesuffix='_small.mp4')
    
    
    boris_file_extract_clips_negativeclass(borisfile, 
                                           behaviors=['Feeding','Resting','Grooming'],
                                           basetime=5,
                                    basepath='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                    basefilesuffix='_small.mp4')   
toc=time.time()-tic    
    



#%% Main processing of OLD  EtholoopData videos, conversion + macaquedetector + crop



#video_original='/data/socialeyes/EtholoopData/2023.05.05_13h45/2023.05.05_13h45_Camera1.avi'


for video_original in ['/data/socialeyes/EtholoopData/2022.08.24_12h28/2022.08.24_12h28_Camera0.avi',
                       '/data/socialeyes/EtholoopData/2022.08.24_12h28/2022.08.24_12h28_Camera1.avi',
                       '/data/socialeyes/EtholoopData/2022.07.08_12h23/2022.07.08_12h23_Camera0.avi',
                       '/data/socialeyes/EtholoopData/2022.07.08_12h23/2022.07.08_12h23_Camera1.avi'
                       ]:


    basedir,videosmall=ffmpeg_video_transformation(video_original,
                                                   rootdir='/data/socialeyes/EthoProcessedVideos')
    
    stride=3
    macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                        weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                        stride=stride,
                        basedir=basedir,
                        source_video=videosmall)
    
    crop_transformed_video(basedir=basedir,
                           source_video=videosmall,
                           stride=stride,
                           unsharp=True)






#%% development zone for Video Dataset preprocessing, nov 2022
import datetime
import pandas as pd

tic=time.time()
annotation_file='/home/amendez/Documents/ML/action_recognition/video_annotations_nov2022.ods'
annotationDF=pd.read_excel(annotation_file)

prev_video=''
for index,row in annotationDF.iterrows():
    video_in=row['video']
    
    #=========== CONVERT WHOLE VIDEO (not just the clips), dont overwrite ====================

    print(f'***** ==== procesing: {video_in} ==== ***** \n')
    prev_video=video_in #update prev_video variable
    #DONT OVERWRITE if video is already there 
    basedir,videosmall=ffmpeg_video_transformation(video_in,
                                                   rootdir='/data/AM_data/monkeybusiness/EthoProcessedVideos_TEMP',
                                                   overwrite=False)
    
    #=========== CUT clips and save them =====================================================  
    
    if (not pd.isnull(row['time_in'])) and (not pd.isna(row['time_in'])):            
        tstartlist=row['time_in'].split(':')# example of time_in 01:20:16.111
        tstart=tstartlist[0]+tstartlist[1]+tstartlist[2].split('.')[0]+tstartlist[2].split('.')[1]
        tendlist=row['time_out'].split(':')
        tend=tendlist[0]+tendlist[1]+tendlist[2].split('.')[0]+tendlist[2].split('.')[1]
        
        #original duration of the clip (maybe the clip needs to be chopped into smaller pieces if it is too large)
        duration=datetime.timedelta(days=0, seconds=float(tendlist[2])-float(tstartlist[2]), 
                     microseconds=0, milliseconds=0, 
                     minutes=int(tendlist[1])-int(tstartlist[1]), 
                     hours=int(tendlist[0])-int(tstartlist[0]), weeks=0).total_seconds()
        
        basetime=10 #desired duration of video segments in seconds
        rest_to_cut=duration #FLOAT, converted to seconds
        cutstartlist=tstartlist #initialization before the loops
        cutstartdt=datetime.datetime.strptime(cutstartlist[0]+':'+cutstartlist[1]+':'+cutstartlist[2], '%H:%M:%S.%f')
        basetimedt=datetime.timedelta(seconds=basetime)
        clips_list=[] #list to hold all clips filenames for further processing
        
        while(rest_to_cut/basetime>2):
            cutenddt=cutstartdt+basetimedt#all of this just to get the ending cutting point without errors         
            cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'
            cutendstr=f'{cutenddt.hour:02d}'+f'{cutenddt.minute:02d}'+f'{cutenddt.second:02d}'+f'{int(cutenddt.microsecond/1000):03d}'
            video_out=os.path.join(os.path.dirname(videosmall),'clips',
                                   cutstartstr+'_'+ cutendstr,
                                   os.path.splitext(os.path.basename(videosmall))[0]+'_'+cutstartstr+'_'+ \
                                   cutendstr+os.path.splitext(videosmall)[1])
            if not os.path.exists(os.path.dirname(video_out)):
                os.makedirs(os.path.dirname(video_out))                
            cut_video_clip(videosmall, video_out,
                           f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}',
                           f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}',
                           overwrite=False)
            #update variables to continue the loop
            rest_to_cut=rest_to_cut-basetime #float
            cutstartdt=cutenddt #use the ending point as the new start cutting point
            print(1,'--->',os.path.basename(video_out))
            clips_list.append(video_out)
            
        if rest_to_cut/basetime<1: 
            #cutstartdt was updated inside the while loop or defined before it if the while was not executed
            cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'
            #make cutenddt the actual end of the clip (tend)
            cutenddt=datetime.datetime.strptime(tendlist[0]+':'+tendlist[1]+':'+tendlist[2], '%H:%M:%S.%f') 
            cutendstr=f'{cutenddt.hour:02d}'+f'{cutenddt.minute:02d}'+f'{cutenddt.second:02d}'+f'{int(cutenddt.microsecond/1000):03d}'
            video_out=os.path.join(os.path.dirname(videosmall),'clips',
                                   cutstartstr+'_'+ cutendstr,
                                   os.path.splitext(os.path.basename(videosmall))[0]+'_'+cutstartstr+'_'+ \
                                   cutendstr+os.path.splitext(videosmall)[1])         
            if not os.path.exists(os.path.dirname(video_out)):
                os.makedirs(os.path.dirname(video_out))       
            cut_video_clip(videosmall, video_out,
                           f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}',
                           f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}',
                           overwrite=False)       
            clips_list.append(video_out)
                
        elif 2>rest_to_cut/basetime>1 and rest_to_cut%basetime<5:
            #cutstartdt was updated inside the while loop or defined before it if the while was not executed
            cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'
            #make cutenddt the actual end of the clip (tend)
            cutenddt=datetime.datetime.strptime(tendlist[0]+':'+tendlist[1]+':'+tendlist[2], '%H:%M:%S.%f') 
            cutendstr=f'{cutenddt.hour:02d}'+f'{cutenddt.minute:02d}'+f'{cutenddt.second:02d}'+f'{int(cutenddt.microsecond/1000):03d}'
            video_out=os.path.join(os.path.dirname(videosmall),'clips',
                                   cutstartstr+'_'+ cutendstr,
                                   os.path.splitext(os.path.basename(videosmall))[0]+'_'+cutstartstr+'_'+ \
                                   cutendstr+os.path.splitext(videosmall)[1])         
            if not os.path.exists(os.path.dirname(video_out)):
                os.makedirs(os.path.dirname(video_out))       
            cut_video_clip(videosmall, video_out,
                           f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}',
                           f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}',
                           overwrite=False)       
            print(2,'--->',os.path.basename(video_out))
            clips_list.append(video_out)

        elif 2>rest_to_cut/basetime>1 and rest_to_cut%basetime>=0.5: #just divide the remaining time by two and voila
            last_piece=rest_to_cut
            for _ in range(2):
                cutenddt=cutstartdt+datetime.timedelta(seconds=last_piece/2)
                cutstartstr=f'{cutstartdt.hour:02d}'+f'{cutstartdt.minute:02d}'+f'{cutstartdt.second:02d}'+f'{int(cutstartdt.microsecond/1000):03d}'
                cutendstr=f'{cutenddt.hour:02d}'+f'{cutenddt.minute:02d}'+f'{cutenddt.second:02d}'+f'{int(cutenddt.microsecond/1000):03d}'
                video_out=os.path.join(os.path.dirname(videosmall),'clips',
                                       cutstartstr+'_'+ cutendstr,
                                       os.path.splitext(os.path.basename(videosmall))[0]+'_'+cutstartstr+'_'+ \
                                       cutendstr+os.path.splitext(videosmall)[1])
                if not os.path.exists(os.path.dirname(video_out)):
                    os.makedirs(os.path.dirname(video_out))  
                cut_video_clip(videosmall, video_out,
                           f'{cutstartdt.hour:02d}:{cutstartdt.minute:02d}:{cutstartdt.second:02d}.{int(cutstartdt.microsecond/1000):03d}',
                           f'{cutenddt.hour:02d}:{cutenddt.minute:02d}:{cutenddt.second:02d}.{int(cutenddt.microsecond/1000):03d}',
                           overwrite=False)
                #update variables to continue the loop
                cutstartdt=cutenddt #use the ending point as the new start cutting point
                print(3,'--->',os.path.basename(video_out))
                clips_list.append(video_out)
                


        #=========== apply Macaquedetector3000 on the CUT clips =================================  
        print(f'clips_list: {clips_list}')
        for clip in clips_list:      
            stride=1
            if not os.path.exists(os.path.join(os.path.dirname(clip),'exp')):
                macaquedetector3000(detect_script='/home/amendez/Documents/clonedrepos/yolov5/detect.py',
                                    weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt',
                                    stride=stride,
                                    basedir=os.path.dirname(clip),
                                    source_video=clip)        
            '''    
            croppedclip=crop_transformed_video(basedir=os.path.dirname(clip),
                                  source_video=clip,
                                  stride=stride,
                                  unsharp=False,
                                  timestamp=False,
                                  stabilize=True,
                                  suffix='_stab')
            '''
            
         #=========== Crop clips according to the detected macaques location ====================                            
            croppedclip=crop_transformed_video_stabilize(basedir=os.path.dirname(clip),
                                  source_video=clip,
                                  stride=stride,
                                  suffix='_s_',
                                  filter_type='gaussian', 
                                  filter_windoworsigma=20)     


            
            
    #========== copy the cropped clip to the appropriate folder    
       
        destination_folder=os.path.join('/data/AM_data/ActionRecognition/datasets/dataset2',
                                              row['macrocategory']) 
                
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        shutil.copy2(croppedclip,os.path.join(destination_folder,os.path.basename(croppedclip)))
            
            
            
toc=time.time()-tic          
            
            
            
            
              
            
            
#%%            

# split the data folder
import splitfolders
splitfolders.ratio('/data/AM_data/ActionRecognition/datasets/dataset2/dataset_all', 
                   output="/data/AM_data/ActionRecognition/datasets/dataset2/", 
                   seed=1337, ratio=(0.8, 0.1,0.1)) 



dataset='dataset2'
vec=['val','train','test']
for partition in vec:
    generate_video_dataset_file(os.path.join('/data/AM_data/ActionRecognition/datasets/',dataset,partition), 
                                os.path.join('/data/AM_data/ActionRecognition/datasets/',dataset, partition+'.txt'),
                                classdir={'chilling':0,'foraging':1,'grooming':2,'moving':3})


os.symlink('/data/AM_data/ActionRecognition/datasets/dataset2','/home/amendez/Documents/clonedrepos/mmaction2/data/dataset2')




'''


#weekend code, halloween edition, create sharpened videos for some already processed cases
tic=time.time()
stride=3
for VIDEO in ['2022.09.23_14h46']: #,'2022.09.07_13h55'
    for CAMERA in ['Camera0','Camera1']:
        basedir=os.path.join('/data/socialeyes/EthoProcessedVideos',VIDEO,CAMERA)
        videosmall=os.path.join(basedir,VIDEO+'_'+CAMERA+'_small.mp4')
        crop_transformed_video(basedir=basedir,
                               source_video=videosmall,
                               stride=stride,
                               unsharp=True)
toc=time.time()-tic        





#weekend code, halloween edition, create sharpened videos for some already processed cases
tic=time.time()
stride=3
for VIDEO in ['2022.09.02_12h12']: #,'2022.08.25_11h59'
    for CAMERA in ['Camera0','Camera1']:
        basedir=os.path.join('/data/socialeyes/EthoProcessedVideos',VIDEO,CAMERA)
        videosmall=os.path.join(basedir,VIDEO+'_'+CAMERA+'_small.mp4')
        crop_transformed_video(basedir=basedir,
                               source_video=videosmall,
                               stride=stride,
                               unsharp=True)
toc=time.time()-tic        
        


'''






#%% ffmpeg downsizing before detection

"""
tic=time.time()

in_video='/data/socialeyes/EtholoopData/2022.08.25_11h59/2022.08.25_11h59_Camera1.avi'

if 'Camera0' in in_video:
    camera='Camera0'
elif 'Camera1' in in_video:
    camera='Camera1'
else:    
    camera='other'
    
basedir=os.path.join('/data/socialeyes/EthoProcessedVideos',os.path.basename(os.path.dirname(in_video)),camera)


if not os.path.exists(basedir):
    os.makedirs(basedir) 

videosmall=os.path.join(basedir,os.path.splitext(os.path.basename(in_video))[0]+'_small.mp4')

ffmpegstring = f'ffmpeg -i "{in_video}" -vf "fps=60, scale=iw/2:ih/2, eq=saturation=1.2,unsharp" -c:v mpeg4 -qscale:v 2 -vsync 1 "{videosmall}"'
subprocess.call(ffmpegstring, shell=True)
toc=time.time()-tic
#-c:v libx264 -crf 10 -preset ultrafast 

#%% MACAQUEDETECTOR3000

#basedir='/data/socialeyes/EthoProcessedVideos/2022.08.25_11h59'
#video='/data/socialeyes/EthoProcessedVideos/2022.08.25_11h59/test640_ultrafast.mp4'
#--------
stride=3
#--------
#os.mkdir(basedir)
weights='/home/amendez/Documents/clonedrepos/yolov5/runs/train/yolov5s_v6_img640/weights/best.pt'
detectstring=f'python /home/amendez/Documents/clonedrepos/yolov5/detect.py --weights {weights} --source {videosmall} --device=0 --line-thickness=3 \
    --iou-thres=0.5 --max-det=2 --save-txt --project {basedir} --vid-stride={stride}'
subprocess.run(detectstring,shell=True)

toc_2=time.time()-tic

#%%
#get the directory where the labels were saved, might be a better way to do it
exp=[a for a in os.listdir(basedir) if os.path.isdir(os.path.join(basedir,a))]
exp.sort()
exp=exp[-1]
txtdirpath=os.path.join(basedir,exp,'labels')

#fix the coordinates indices (according to the stride used in the monkeydetector), it takes a while
txtdirpathfixed=fix_detected_filenames_stride(txtdirpath, stride, videosmall)

#---------------------------------- load the video to be cropped with cv2
cap = cv2.VideoCapture(videosmall)
#----------------------------------
numberofframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


#get the numpy array holding the coordinates "xl,yl,xr,yr,center[0],center[1]"
coords=get_crop_coord_list(txtdirpathfixed,videosmall,numberofframes)

outvideofile=os.path.join(basedir,os.path.splitext(os.path.basename(videosmall))[0]+'_CROP.mp4')
out = cv2.VideoWriter(outvideofile,cv2.VideoWriter_fourcc(*'mp4v'), 60, (width,width))
kernel=np.array([[-1,-1,-1],[-1,6,-1],[-1,-1,-1]]) #in case you want to apply it to each frame

#parameters to print the HHMMSSSS into the frame
font = cv2.FONT_HERSHEY_COMPLEX
org = (10, 530) #x,y
color = (0, 0, 255) #red on BGR    
fontScale = 1
thickness = 2


#frame_list=[]
frame_idx=0
bar=progressbar.ProgressBar(max_value=numberofframes, redirect_stdout=True) #this is the class ProgressBar not the function progressbar, check the documentation
while(1):
    ret, frame = cap.read() #Read Frame
    if not ret:
        print('No frames grabbed!')
        break
    framecrop=crop_frame(frame,coords[frame_idx,:])  

    #framecrop=cv2.filter2D(framecrop,-1,kernel) #applying a sharpening mask
    #write the timestamp 
    _,hh,mm,ss,=convert_from_ms(cap.get(cv2.CAP_PROP_POS_MSEC))#get timestamp
    hhmmss=f'{hh:02d}:{mm:02d}:{ss:05.3f}'
    _=cv2.putText(framecrop, hhmmss, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA,bottomLeftOrigin=False)
        
    out.write(framecrop)
    #frame_list.append(framecrop.astype('uint8'))
    bar.update(frame_idx)
    frame_idx+=1
    #if frame_idx%5000==0:
        #break
        #print(f'\nframe: {frame_idx} of {numberofframes}')
cap.release()
out.release()

toc=time.time()-tic
"""
