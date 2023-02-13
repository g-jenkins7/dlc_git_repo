# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:45:19 2022

@author: George
"""

#%% Dependencies
import pickle
import numpy as np
import os
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import pdb 
import subprocess
import re
import cv2
import scipy as sp
from scipy.signal import find_peaks
from scipy import interpolate

class DLC_data_struc:
    def __init__(self,tag):
        self.tag = tag
        self.animal = tag[3:5]
        self.session = tag[-3:]
        self.dlc_data_preproc = []
        self.dlc_data = []
        self.double_side = None
        self.err_times_med  = []
        self.trial_start_times_med = []
        self.sess_start_time_med  = []
        self.cut_start = []
        self.cut_end = []
        self.trial_start_frames = []
        self.trial_end_frames = []
        self.dlc_data_norm = []
        self.track_by_trial = [] 
        self.track_by_trial_norm = [] 
        self.err_frames = []
        self.box_medians = []
        self.box_norm_medians = []
        self.trial_data= [] 
        self.bodyparts =[]
        self.trial_succ_times_med = []
        self.trial_succ_times_frames = []
        self.frame_time_in_sec = None
        self.med_vid_diff = None
        self.error_match_data = [] 
        self.err_range = None
        self.dlc_data_scaled = [] 
        self.track_by_trial_scaled = []
        self.error_alignment_setting = {'start_frame': 400 ,
                                       'spike_threshold': 5, # remove high spikes in brightness
                                       'rebound_threshold': -5, #remove rebounds 
                                       'peak_scale_factor':  1.53,
                                       'find_peaks_distance':  110}


def get_frame_times(input_name, output_file):
    print('~.~.~ working on ' + input_name + ' ~.~.~')
    cmd = "ffmpeg -i {0} -f null -vf showinfo - 2> {1}".format(input_name, output_file)
    p=subprocess.check_output(cmd, shell=True) # get vid infro from ffmpeg 
    
    # extract each frame time and append to list 
    frame_times = []
    with open(output_file) as f:
        for line in f:
            if 'pts_time' in line:
                line_split = line.split()
                loc = [line_split.index(x) for x in line_split if 'pts_time' in x]
                frame_times.append(line_split[loc[0]].split(':')[1])
    return frame_times

def get_brightness(input_name, vid_directory):
    cap = cv2.VideoCapture(vid_directory + '/' + input_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_brightness= np.zeros(length)

    i = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(frame_hsv)
        avg_brightness[i] = np.mean(v)
        i += 1       
    return avg_brightness

                                       
def process_videos(vid_directory,exp, old_frame_dict = None, old_brightness_dict = None, update = False):
    """
    this function takes input videos and calculates the time of each frame in seconds
    and the average brightness of each frame for aligning to errors    
    data for each video is stored under a tag in the form "ratxx_xxx" which is preserved across dlc analyis functions 
    Parameters
    ----------
    vid_directory : location of .mp4 or .avi video files 

    Returns
    -------
    all_frame_times : dictionary with an array of frame times for each video
        
    all_avg_brightness : dictionary with an array of avg brightness vals for each video

    """
    # gets frame times using ffmpeg and average brightness per frame using cv2    
    count = 1
    os.chdir(vid_directory)
    video_files = os.listdir(vid_directory)
    video_files  = [x for x in video_files if '.mp4' in x]
    all_frame_times = {} 
    all_avg_brightness = {}
    for file in video_files: 
        input_name = file # specify video
        rat_ind = input_name.find('rat')
        if exp == 'ecb':
            file_tag = input_name[rat_ind: -10]
            if '10' in file_tag:
                file_tag = input_name[rat_ind:-10]
            else:
                file_tag = input_name[rat_ind: -9]
        else:
            if 'bmi' in input_name:
                 file_tag =  input_name[rat_ind: rat_ind+9] +'bmi'
            elif 'sal' in input_name:
                file_tag =  input_name[rat_ind: rat_ind+9] +'sal'
                 
        output_file = file_tag + 'vid_info.txt' # name output file 
        if update == True:
            if file_tag  not in old_frame_dict.keys():                
               all_frame_times[file_tag] = get_frame_times(input_name, output_file)
               all_avg_brightness[file_tag] = get_brightness(input_name, vid_directory)
               print(count)
               count += 1
        else:
            all_frame_times[file_tag] = get_frame_times(input_name, output_file)
            all_avg_brightness[file_tag] = get_brightness(input_name, vid_directory)
            print(count)
            count += 1
    if update == True:
        all_frame_times.update(old_frame_dict)
        all_avg_brightness.update(old_brightness_dict)
    return all_frame_times, all_avg_brightness

def get_beh_dlc_data(session_list, subject_list,all_beh_data,all_trial_data,dlc_file_path):
    """
    This functions retrieves key features of behavioural data and dlc tracking data for each session -  then 
    stores them all as attributes under the 'DLC data struc' 

    Parameters
    ----------
    session_list : list of sessions to investigate
    subject_list : list of subjects to investiage
    all_beh_data : dict with raw med-pc data for each session
    all_trial_data : post beh analysis dataframe with every trial from every animal/session included
    dlc_file_path : path where dlc .h5 files are located

    Returns
    -------
    all_data : dictionary containing a DLC_data_struc for each video/behavioural session 

    """    
    all_data = {} 
    for session in session_list:
        for animal in subject_list: 
            print(animal)
            print(session)
            tag = 'rat' + animal + '_' + session
            all_data[tag] = DLC_data_struc(tag)
            beh_data = all_beh_data[session][animal]['0']
            trial_data = all_trial_data.query('session == @session and animal == @animal')
            all_data[tag].trial_data = trial_data
            #get times of events 
            
            df_list        = beh_data.keys()
            df_sizes       = [len(beh_data[x]) for x in df_list]
            df_events_name = list(df_list)[max((v,i) for i,v in enumerate(df_sizes))[1]]
            events_data  = beh_data[df_events_name]
            
            name_df_U   = [x for x in list(beh_data.keys()) if 'U' in beh_data[x].columns][0]
            double_side = list(beh_data[name_df_U]['U'])[-1]
            all_data[tag].double_side = double_side
            
            #get error times
            error_start =  events_data[events_data.E == 503]
            ix_error_start = error_start.index
            error_times = events_data['T'][ix_error_start]
            all_data[tag].err_times_med = error_times
            
            
            #get trial start times
            trial_start =  events_data[events_data['E'] == 103]
            ix_trial_start = trial_start.index
            trial_start_times = events_data['T'][ix_trial_start]

            all_data[tag].trial_start_times_med =trial_start_times
            
            trial_succ = events_data[(events_data['E'] == 3022 ) | (events_data['E'] == 3012 ) | (events_data['E'] == 3032)]
            trial_succ_times = trial_start_times.copy()
            trial_succ_times = trial_succ_times.reset_index()
            trial_succ_times = trial_succ_times.drop('index',axis =1)
            for t in trial_succ['T']:
                col = trial_succ_times - t
                smallest_negative = col[col['T'] < 0].max()
                idx = col.query('T == @smallest_negative[0]').index[0]
                trial_succ_times.loc[idx,'trial_succ_time'] = t
            trial_succ_times = trial_succ_times.drop('T',axis =1)
            all_data[tag].trial_succ_times_med  =trial_succ_times
                        
            
            
            #get session start time
            sess_start = events_data[events_data['E'] == 100]
            ix_sess_start = sess_start.index
            sess_start_time = events_data['T'][ix_sess_start]
            all_data[tag].sess_start_time_med = sess_start_time
            
            #load dlc file            
            os.chdir(dlc_file_path)
            files = os.listdir(dlc_file_path)
            dlc_files = [x for x in files if '.h5' in x]
            tag = 'rat' + animal + '_' + session   
            dlc_file = [x for x in dlc_files if 'rat' + animal in x and session in x] 
            if len(dlc_file) == 0:
                print(tag + ' no analysed video found \
                      --------------------------------')
            elif len(dlc_file) > 1:
                print(tag + ' mulitple videos found!!! \
                     --------------------------------') 
            else:           
                dlc_data_preproc = pd.read_hdf(dlc_file[0])
                dlc_data_preproc.columns = dlc_data_preproc.columns.droplevel() # drop 1st level of M_idx
                all_data[tag].bodyparts = list(dlc_data_preproc.columns.get_level_values(0).unique()) # get bodyparts list
                all_data[tag].dlc_data_preproc = dlc_data_preproc
            
    return all_data


# 
#     return all_data, confidence_prop

def _remove_short_sections(x, min_len):
    xc = np.copy(x)
    ps = np.nan
    for i, s in enumerate(x):
        if np.isfinite(s) and np.isnan(ps):
            run_start=i
        elif np.isnan(s) and np.isfinite(ps):
            run_end = i
            if (run_end-run_start)<min_len:
                xc[run_start:run_end] = np.nan
        ps = s 
    return xc

def interpolate_dlc_data(all_data, intp_method,lik_threshold =0.9, jump_threshold=10, min_len=5): 
    '''
    Parameters
    ----------
    all_data : dict
        dict of dlc_data_struc for each session.
    intp_method : int 
        method of interpolation .
    lik_threshold: int
        anything below this form DLC is discarded
    jump_threshold: int
        diff in coords > jump_threshold * sd are discarded
    min_len: int
        miniumum len of isolated point, any less are discarded
    Returns
    -------
    all_data : TYPE
        dict of dlc_data_struc for each session.
    valid_points : df
        perc of valid points for each session analysed.

    '''
    valid_points = pd.DataFrame()
    for tag in all_data.keys():
        print(tag)
        d = all_data[tag].dlc_data_preproc['head']
        
        x_clean = d.x.copy()
        y_clean = d.y.copy()
        lik = d.likelihood
        lik_threshold =0.9
        jump_threshold=10
        min_len=5
        
        #remove lower than lik threshold 
        x_clean[lik<lik_threshold] = np.nan
        y_clean[lik<lik_threshold] = np.nan
        
        #remove jumps
        dx = np.diff(x_clean)
        dy = np.diff(y_clean)
        dd = np.sqrt(dx**2+dy**2)
        sd_est = np.diff(np.nanpercentile(dd, (16,84)))/2 # Standard deviation of dd estimated from percentiles.
        jump_inds = np.where(np.abs(dd)>(jump_threshold*sd_est))[0]
        x_clean[jump_inds] = np.nan
        x_clean[jump_inds+1] = np.nan
        y_clean[jump_inds] = np.nan
        y_clean[jump_inds+1] = np.nan
        
        #remove isolated points
        x_clean = _remove_short_sections(x_clean,min_len)
        y_clean = _remove_short_sections(y_clean,min_len)
        
        print(f'Fraction valid points    X: {np.mean(np.isfinite(x_clean)) :.2f}, '
                                      f' Y: {np.mean(np.isfinite(y_clean)) :.2f}')
        valid_points.loc[tag,'perc_valid_points'] = np.mean(np.isfinite(x_clean))
        #interpolate
        if intp_method == 1:
            x_clean = pd.DataFrame(x_clean)
            y_clean = pd.DataFrame(y_clean)
            
            x_clean = x_clean.interpolate(limit_direction = 'both', limit = 10)
            y_clean = y_clean.interpolate(limit_direction = 'both', limit = 10)
            x_clean = x_clean.values
            y_clean = y_clean.values
    
        elif intp_method ==2:
            x_clean = nan_fill(x_clean)
            y_clean = nan_fill(y_clean)
        
        all_data[tag].dlc_data = all_data[tag].dlc_data_preproc.copy()
        all_data[tag].dlc_data[('head','x')] = x_clean
        all_data[tag].dlc_data[('head','y')] = y_clean
    return all_data, valid_points
    


def nan_fill(x):
    inds = np.arange(x.shape[0])
    good = np.where(np.isfinite(x))
    f = interpolate.interp1d(inds[good[0]], x[good], bounds_error =False, fill_value=np.nanmedian(x))
    B = np.where(np.isfinite(x),x,f(inds))
    return B 

def find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,all_data, plot_peaks,subject_list,session_list, check_err=False):
    '''

    Parameters
    ----------
    all_frame_times : dict
        dict of frame times for each session .
    all_avg_brightness : dict
        dict of avg brightness values extracted from each video .
    all_data : dict
        dict of dlc data struc.
    plot_peaks : str
        'y' to plot and 'n' to not .
    subject_list : list
        list of subjects .
    session_list : list
        lsit of sessions.
    check_err : bool, optional
        check errors with plots. The default is False.

    Returns
    -------
    all_peaks : dict
        dict of luminance peaks for each session.
    all_onsets : dict
        dict of trial onset times for each session .
    mismatch_list : LIST
        list of files that have a mismatch between the num peaks 
        and num recorded errors.

    '''
    all_peaks ={} 
    all_onsets={}
    counter = 0 
    mismatch_list = []

    for sub in subject_list: 
        for session in session_list:
            for tag in list(all_data.keys()):
                print(tag)
                if sub in tag and session in tag:
                    counter += 1
                    
                    onsets, peaks, mismatch_list =  get_peaks(all_frame_times,all_avg_brightness
                                                              ,all_data, plot_peaks,
                                                              tag, mismatch_list)
                                                                                 

                    all_onsets[tag] = onsets
                    all_peaks[tag] = peaks
    return all_peaks, all_onsets,mismatch_list

    

def get_peaks(all_frame_times,all_avg_brightness, all_data, plot_peaks, tag, mismatch_list, check_err = False):
           #NEED TO DO THOROUGH CHECK ON OTHER VIDOES 
    
    
    # throwing away first 400 frames as thats when box still open usually? might ened to modify this value or include sanity check
    norm_brightness = all_avg_brightness[tag] - np.mean(all_avg_brightness[tag]) 
    norm_brightness = norm_brightness[all_data[tag].error_alignment_setting['start_frame']: all_data[tag].error_alignment_setting['end_frame']]
    norm_brightness = norm_brightness + all_data[tag].error_alignment_setting['upshift']
    
   
    #chop off spikes
    despk_brightness = norm_brightness.copy()
    d_brightness = np.diff(despk_brightness)
    over_idx = np.where(d_brightness > all_data[tag].error_alignment_setting['spike_threshold']) # original is 5 
   
    #chop off rebound??
    under_idx = np.where(d_brightness < all_data[tag].error_alignment_setting['rebound_threshold']) 
    for e in over_idx[0]:
        despk_brightness[e:e+10] = np.nan
    despk_brightness = nan_fill(despk_brightness)
    
    for i in under_idx[0]:
        despk_brightness[i:i+10] = np.nan
    despk_brightness = nan_fill(despk_brightness)

    # taking mid section 
    session_brightness = despk_brightness[10000 + all_data[tag].error_alignment_setting['brightness_sample_offset']:
                                          20000 + all_data[tag].error_alignment_setting['brightness_sample_offset']]
    #set threshold for peak     
    min_peak_prom = session_brightness.max() / all_data[tag].error_alignment_setting['peak_scale_factor'] # each file has a min peak prominence scaled by size of its peaks 
    
    #get peaks
    peak, _  = find_peaks(despk_brightness , distance = all_data[tag].error_alignment_setting['find_peaks_distance'])
    proms,_,_ = sp.signal.peak_prominences(despk_brightness, wlen=None, peaks = peak)
    
    #filter_peaks
    min_idx = proms > min_peak_prom
    filtered_peaks = peak[min_idx]
    all_proms = proms[min_idx]
    no_spike_idx =all_proms < (all_proms.mean() + (all_proms.std() *3.9)) # removing huge birghtness spike form open dorr in some files
    filtered_peaks = filtered_peaks[no_spike_idx]

    
    # finding onset of peaks - eg where error light turns on
    filtered_onsets = [] 
    for p in range (0,len(filtered_peaks)): 
        if p == 0: # if first peak in recordingh 
            if filtered_peaks[p] > 250:
                error_section = despk_brightness[filtered_peaks[p] -250 : filtered_peaks[p]].copy() # get section with error in 
                error_section_diff = np.diff(error_section) # find sharpest increase in brightness 
                error_onset = np.where(error_section_diff == error_section_diff.max())[0][0]  # get index of that increase -time of onset        
                filtered_onsets = np.append(filtered_onsets, filtered_peaks[p] +(error_onset - 250))   
            else: # if less than 250 frames from start then jsut take first frame 
                error_section = despk_brightness[0 : filtered_peaks[p]].copy()      
                error_section_diff = np.diff(error_section)
                error_onset =np.where(error_section_diff == error_section_diff.max())[0][0]
                filtered_onsets = np.append(filtered_onsets, error_onset)
                
        elif filtered_peaks[p] - filtered_peaks[p-1] < 250: #if less than 250 frames after last peak - take section just after last peaks
            error_section = despk_brightness[filtered_peaks[p] -(filtered_peaks[p] - filtered_peaks[p-1]) : filtered_peaks[p]].copy()
            error_section_diff = np.diff(error_section)
            error_onset =np.where(error_section_diff == error_section_diff.max())[0][0]
            filtered_onsets = np.append(filtered_onsets, filtered_peaks[p] +(error_onset - (filtered_peaks[p] - filtered_peaks[p-1])))
            
        else: # otheriwse take 250 frames before 
            error_section = despk_brightness[filtered_peaks[p] -250 : filtered_peaks[p]].copy()    
            error_section_diff = np.diff(error_section)
            error_onset = np.where(error_section_diff == error_section_diff.max())[0][0]
            filtered_onsets = np.append(filtered_onsets, filtered_peaks[p] +(error_onset - 250))
                
     
        
    filtered_onsets = [int(x) for x in filtered_onsets] #
    filtered_onsets = np.delete(filtered_onsets, np.argwhere(np.ediff1d(filtered_onsets) <= 150) + 1) # remove any peaks that are closer together than should be possible 
        
    if plot_peaks != 'n': #plot brights, peaks and onsets
        plt.figure(figsize = (18,7))
        plt.plot(despk_brightness)
        plt.title(tag)
        plotted_peaks = plt.plot(filtered_peaks, (despk_brightness)[filtered_peaks], '*')
        plotted_onsets = plt.plot(filtered_onsets, (despk_brightness)[filtered_onsets], '*', color ='r')
        
        if check_err == True: # need medpc errors - plot those too 
            yvals = np.zeros(len(all_data[tag].err_frames))
            yvals = yvals + despk_brightness[filtered_peaks].max() + despk_brightness[filtered_peaks].max()*0.1
            plotted_med_errs = plt.plot(all_data[tag].err_frames- all_data[tag].error_alignment_setting['start_frame'], 
                                        yvals, '*','g')
            plt.ylabel('Normalised brightness')
            plt.xlabel('frame_number')
         
            print('~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.')
            print(tag + 'med error vs brightness lag')
            print(filtered_peaks[-1] - all_data[tag].err_frames.iloc[-1])
            
        plt.show()
        
    # min peaks is a leftover from lauras code - dont think we need this
    if len(filtered_onsets) != len(all_data[tag].err_times_med):
        print(tag)
        print('ERROR, MED to DLC mismatch')
        print('filt_onsets = ' +  str(len(filtered_onsets))+ ' err_times_med ' + str(len(all_data[tag].err_times_med)))
        print('~~~~~~~~~~~~~~~~~')
        mismatch_list = np.append(mismatch_list, tag)
    elif plot_peaks == 'err_only':
        plt.close()
    filtered_onsets = np.array(filtered_onsets) + all_data[tag].error_alignment_setting['start_frame']
    filtered_peaks = filtered_peaks + all_data[tag].error_alignment_setting['start_frame']
    return filtered_onsets, filtered_peaks, mismatch_list    
    
def get_trial_frames(all_data,all_onsets,all_frame_times,period):
    """
    

    Parameters
    ----------
    all_data : TYPE Dict
        DESCRIPTION. contains DLC_data_strucs for each session
        
    all_peaks : dictionary with the brightness peaks for each video
    
    all_frame_times : dict
        DESCRIPTION. contains array of frame times in secs for each sessions

    Returns
    -------
    all_data :- now updated to include trial_start_frames and trial_end_frames
    mismatched files: list of error files to check 

    """
    mismatched_files = []
    for tag in all_onsets.keys():#['rat09_10mg']:#.keys():
       # fr_in_s = [float(x) for x in all_frame_times[tag]]
        #frame_in_sec = np.diff(fr_in_s).mean() # = np.diff(fr_in_s).mean() # seems like frame times regular so far... getting time inbetween frames
        frame_in_sec = 0.04 # hard set to 0.04 even if ffmpeg says otherwise
        all_data[tag].frame_in_sec = frame_in_sec
        errors = pd.DataFrame()
        errors.loc[:,'err_times_med'] = all_data[tag].err_times_med/100 # convert to  sec
        errors = errors.reset_index(drop = True)
        print(tag)
        errors.loc[:,'err_times_vid'] = pd.DataFrame((all_onsets[tag])*frame_in_sec )
        errors.loc[:,'med_vid_diff']  = errors.err_times_vid - errors.err_times_med
        errors = errors.drop(len(errors)-1)
        error_range = errors.med_vid_diff.max() - errors.med_vid_diff.min() 
        all_data[tag].error_range =error_range
        
        if error_range > 0.5:
            print(tag)
            print('video to med error mismatch')
            print(error_range)
            mismatched_files.append(tag)
             
        all_data[tag].error_match_data = errors
        
        avg_med_vid_diff = errors.med_vid_diff.mean()  
        all_data[tag].avg_med_vid_diff = avg_med_vid_diff

        
        trial_start_times_vid = (all_data[tag].trial_start_times_med/100) + avg_med_vid_diff
        trial_succ_times_vid= (all_data[tag].trial_succ_times_med/100) + avg_med_vid_diff
        
        
        trial_start_frames = trial_start_times_vid/frame_in_sec # conver to frames
        trial_succ_times_frames = trial_succ_times_vid/frame_in_sec
        trial_end_frames = trial_start_frames + (period/frame_in_sec) # take x seconds after trial start 
        
        trial_start_frames = trial_start_frames.apply(np.floor)#round to closest 
        all_data[tag].trial_start_frames = trial_start_frames.astype(int)
        
        trial_end_frames = trial_end_frames.apply(np.floor)#round to closest  
        all_data[tag].trial_end_frames = trial_end_frames.astype(int)
        
        trial_succ_times_frames = trial_succ_times_frames.apply(np.ceil)
        #trial_succ_times_frames = trial_succ_times_frames.dropna()
        all_data[tag].trial_succ_times_frames = trial_succ_times_frames.astype('Int64')
        all_data[tag].err_frames = ((all_data[tag].err_times_med/100)+avg_med_vid_diff)/frame_in_sec
        
        
    return  all_data, mismatched_files



def get_distances(all_data,avg_all_norm_medians):
    distances = pd.DataFrame()
    for x in all_data.keys():
        if all_data[x].box_norm_medians:
            
            distances.loc[x,'lever_distance_x'] = all_data[x].box_norm_medians['l_lever_x'] - all_data[x].box_norm_medians['r_lever_x'] 
            distances.loc[x,'box_distance_y'] = ((all_data[x].box_norm_medians['boxtopleft_y'] - all_data[x].box_norm_medians['boxbottomleft_y']) 
                                                        + (all_data[x].box_norm_medians['boxtopright_y'] - all_data[x].box_norm_medians['boxbottomright_y']))/2


    avg_lever_distance_x = avg_all_norm_medians.l_lever_x -    avg_all_norm_medians.r_lever_x

    avg_box_distance_y = ((avg_all_norm_medians.boxtopleft_y - avg_all_norm_medians.boxbottomleft_y)
                            + (avg_all_norm_medians.boxtopright_y - avg_all_norm_medians.boxbottomright_y))/2
                                                                                                                        

    distances['scale_factor_x'] =    avg_lever_distance_x/ distances.lever_distance_x        
    distances['scale_factor_y'] =    avg_box_distance_y/ distances.box_distance_y        
    distances['files'] = list(distances.index)
    
    return distances
    
       
def scale_dlc_data(D, traj_part, distances):
    """
    scale dlc tracking data 

    Parameters
    ----------
    D : DLC data class
        DESCRIPTION.
    traj_part : TYPE
        DESCRIPTION.
    distances : df
        box feature distances for scaling traj .

    Returns
    -------
    dlc_data_scaled : df
        dflc_nrom data for D scaled.

    """
    dlc_data_scaled = D.dlc_data_norm.copy()
    dlc_data_scaled.loc[:,(traj_part,'x')] = (D.dlc_data_norm['head']['x'] * distances.query('files == @D.tag').scale_factor_x[0])
    dlc_data_scaled.loc[:,(traj_part,'y')] = (D.dlc_data_norm['head']['y'] * distances.query('files == @D.tag').scale_factor_y[0])
    return dlc_data_scaled

def get_medians(D, norm= False):
    """
    calculation median x,y coordinates from box features

    Parameters
    ----------
    D : DLC data class
        DESCRIPTION.
    norm : bool, optional
        calc from nomalized datae

    Returns
    -------
    medians : df
        box feature median coordinates.

    """
    box_features = [
        'poke',
        'l_lever',
        'r_lever',
        'l_foodmag',
        'r_foodmag',
        'boxtopleft',
        'boxtopright',
        'boxbottomleft',
        'boxbottomright']  
    medians = {}
    if norm == True:
        for part in box_features:
            medians[part + '_x'] = D.dlc_data_norm[part].x.median()
            medians[part + '_y'] = D.dlc_data_norm[part].y.median()
    else:
        for part in box_features:
            medians[part + '_x'] = D.dlc_data[part].x.median()
            medians[part + '_y'] = D.dlc_data[part].y.median()
    return medians

def normalise_dlc_data(D):
    """
    normalisaes dlc_data by subtracint nosepoke coordinates

    Parameters
    ----------
    D : DLC data class

    Returns
    -------
    norm_data : df
        

    """
    norm_data = D.dlc_data.copy()
    for part in D.bodyparts:
        norm_data.loc[:,(part,'x')]=  D.dlc_data[part].x - D.box_medians['poke_x']
        norm_data.loc[:,(part,'y')]=  D.dlc_data[part].y - D.box_medians['poke_y']
    setattr(D,'dlc_data_norm',norm_data)
    return norm_data



def restrict_trajectories(D, data, traj_part): 
    """
    restricts trackign data to no futher on y axis than food magazine

    Parameters
    ----------
    D : DLC data class
        .
    data : df
        dlc tracking data attrtibute of D . 
     traj_part : str
         body part to track.

    Returns
    -------
    data : df
        dlc tracking data now restricted.

    """
    # SET ANY YVLAS PAST THE FOOD MAG TO THE same y as the food mag
    trajy = data[traj_part].y.copy()
    trajy[trajy < 0] = 0
    trajy[trajy > (D.box_norm_medians['r_foodmag_y'] + 
                   D.box_norm_medians['l_foodmag_y'])/2 ] = (D.box_norm_medians['r_foodmag_y'] +
                                                       D.box_norm_medians['l_foodmag_y'])/2
    data.loc[:,(traj_part,'y')] = trajy
    return data
                                         
def track_trials(D, data, traj_part, extra_time):
    """
    breaks tracking data into trials 

    Parameters
    ----------
    D : DLC data class
        .
    data : df
        dlc tracking data attrtibute of D . 
        eg "dlc_data_scaled" or "dlc_data_norm"
    extra_time : float
        time in s after succesful trial med event to track for..

    Returns
    -------
    tracked_trials. dict
    

    """

    tracked_trials  ={}
    for trial_no in range(0,len(D.trial_start_frames)):
        print(trial_no)
        if D.trial_start_frames.iloc[trial_no] < data.index[-1]:
            if D.trial_succ_times_med.empty or pd.isnull(D.trial_succ_times_med.iloc[trial_no])[0]:              #if failed trial 
                tracked_trials[trial_no] = data[D.trial_start_frames.iloc[trial_no]: 
                                               D.trial_end_frames.iloc[trial_no]]
               # print(tracked_trials)
            else:
                trial_succ_tracking = get_succ_trial_end(D, data,  
                                                         trial_no, 
                                                         traj_part, extra_time)
                tracked_trials[trial_no] = trial_succ_tracking
    return tracked_trials
  
def get_succ_trial_end(D, data,trial_no, traj_part, extra_time):
    """
    select when to stop tracking on successful trials. 
    Either as defined by "extra time" if they enter the food mag roi in that time
    or the same as a failed trial if not 

    Parameters
    ----------
    D : DLC data class
        .
    data : df
        dlc tracking data attrtibute of D . 
        eg "dlc_data_scaled" or "dlc_data_norm"
    traj_part : str
        body part to track.
    trial_no : int
    .
    extra_time : float
        time in s after succesful trial med event to track for..
    Returns
    -------
    succ_tracking : df
        subset of data df for this trial .

    """
    #define ROI around food mag
    print('succ')
    lx = D.box_norm_medians['l_foodmag_x']
    ly = D.box_norm_medians['l_foodmag_y']
    ry = D.box_norm_medians['r_foodmag_y']
    rx = D.box_norm_medians['r_foodmag_x']
    
    roi_lx = lx - (np.diff([lx,rx])[0]/2)
    roi_ly = ly- (np.diff([lx,rx])[0]/2)
    roi_x_range = [roi_lx, roi_lx +  np.diff([lx,rx])[0]*2]
    roi_y_range = [roi_ly,roi_ly + np.diff([roi_ly,ly])[0]]
    
    trial_data = data[D.trial_start_frames.iloc[trial_no]: D.trial_end_frames.iloc[trial_no]+100][traj_part]
    in_roi = trial_data.query('x > @roi_x_range[0] and x < @roi_x_range[1] and y > @roi_y_range[0] and y < @roi_y_range[1]' )
    succ_time = D.trial_succ_times_frames.iloc[trial_no]
    if in_roi.empty: #if they dont make to food mag within time period just do same as for fail trials
        succ_tracking = data[ D.trial_start_frames.iloc[trial_no]: D.trial_end_frames.iloc[trial_no]]
    else:
        first_mag_entry = in_roi[in_roi.index > succ_time[0]]# .iloc[0].name
        if first_mag_entry.empty:
            succ_tracking = data[ D.trial_start_frames.iloc[trial_no]: D.trial_end_frames.iloc[trial_no]]
        else:
            succ_tracking = data[D.trial_start_frames.iloc[trial_no]: first_mag_entry.iloc[0].name + int(extra_time)]
        
    return succ_tracking

def normalise_and_scale(all_data,all_frame_times,excluded_files):
    """
    parent function - normalises dlc data, scales and calcs

    Parameters
    ----------
    all_data : dict
        dict of dlc data structs.
    all_frame_times : dict
        dict of frame times.
    excluded_files : list 
        list of files to be excluded from analysis.

    Returns
    -------
   all_data : dict
       dict of dlc data struct - updated with info
    distances : df
        box feature distances for scaling traj .

    """
    
    for tag in all_data.keys():
        #if tag not in excluded_files:
        # get median of box features across trials 
        #N frames after succ to track succ trials 
        print(tag)
        box_medians = get_medians(all_data[tag]) # get medians for box features 
        setattr(all_data[tag], 'box_medians', box_medians)
        
        dlc_data_norm = normalise_dlc_data(all_data[tag]) # subtract poke median to normalise data
        setattr(all_data[tag], 'dlc_data_norm', dlc_data_norm)
        
        box_norm_medians = get_medians(all_data[tag], norm = True) #get medians of normalised data 
        setattr(all_data[tag], 'box_norm_medians', box_norm_medians)
        
    all_avg_norm_medians = get_avg_norm_medians(all_data) # avg medians for plotting     
    distances = get_distances(all_data,all_avg_norm_medians)
    
    return all_data,distances

def track_all(all_data ,excluded_files,distances, succ_extra_time = 0.75, traj_part = 'head', restrict_traj = True):
    """
    parent function for scaling trajectories and tracking each trial 
    
    Parameters
    ----------
    all_data : dict
        dict of dlc data structs.
   excluded_files : list 
       list of files to be excluded from analysis.
    distances : df
         box feature distances for scaling traj .
    traj_part : stR, optional
        body part to track . default is head
    succ_extra_time : float, optional
        time in s after succesful trial med event to track for.
             default is 0.75s
    restrict_traj : BOOL, optional
        restrict traj up to food mag in y direction.
        The default is True.

    Returns
    -------
    all_data : dict
        dict of dlc data struct - updated with info

    """
    all_track_by_trial = {}
    frame_time_in_sec = 0.04# = float(all_frame_times[tag][1])
    extra_time = np.floor(succ_extra_time/frame_time_in_sec)
    for tag in all_data.keys():
        if tag not in excluded_files and tag != 'rat01_14_sal' and tag != 'rat01_14_sal':
            print(tag)
            dlc_data_scaled = scale_dlc_data(all_data[tag], traj_part, distances)
            setattr(all_data[tag], 'dlc_data_scaled', dlc_data_scaled)

            if restrict_traj:
                all_data[tag].dlc_data_norm = restrict_trajectories(all_data[tag], all_data[tag].dlc_data_norm, traj_part)  
                all_data[tag].dlc_data_scaled = restrict_trajectories(all_data[tag], all_data[tag].dlc_data_scaled, traj_part)   
            
            
            track_by_trial =  track_trials(all_data[tag], all_data[tag].dlc_data, 
                                           traj_part, extra_time)  
            setattr(all_data[tag], 'track_by_trial', track_by_trial)
            track_by_trial_norm = track_trials(all_data[tag], all_data[tag].dlc_data_norm, 
                                               traj_part, extra_time) 
            setattr(all_data[tag], 'track_by_trial_norm', track_by_trial_norm)
            track_by_trial_scaled = track_trials(all_data[tag], all_data[tag].dlc_data_scaled,
                                                 traj_part, extra_time)
            setattr(all_data[tag], 'track_by_trial_scaled', track_by_trial_scaled)
       

    return all_data, 
        
        
        
        
        
def plotbox(ax,norm_medians):
    """
    plots box features 

    Parameters
    ----------
    ax : TYPE axis
        DESCRIPTION. ax to plot on 
    norm_medians : df 
        DESCRIPTION. df of normalised medians for file of interest

    Returns
    -------
    None.

    """
    ax.scatter(norm_medians['poke_x'],norm_medians['poke_y'], c='cornflowerblue')
    ax.scatter(norm_medians['l_lever_x'],norm_medians['l_lever_y'], c='forestgreen')
    ax.scatter(norm_medians['r_lever_x'],norm_medians['r_lever_y'], c='forestgreen')
    
    ax.scatter(norm_medians['l_foodmag_x'],norm_medians['l_foodmag_y'], c='gold')
    ax.scatter(norm_medians['r_foodmag_x'],norm_medians['r_foodmag_y'], c='gold')
    
    ax.scatter(norm_medians['boxtopleft_x'],norm_medians['boxtopleft_y'], c='firebrick')
    ax.scatter(norm_medians['boxtopright_x'],norm_medians['boxtopright_y'], c='firebrick')
    ax.scatter(norm_medians['boxbottomleft_x'],norm_medians['boxbottomleft_y'], c='firebrick')
    ax.scatter(norm_medians['boxbottomright_x'],norm_medians['boxbottomright_y'], c='firebrick')
    plt.show()
    return ax 



def frame_checker(tag_list,vid_directory,frame_list,frame_range,all_data,method,manual=False):
    """
    run through frames before and after event (eg trial start times or error times)

    Parameters
    ----------
    tag_list : TYPE list
        DESCRIPTION. list of tags (files) to check frames of
    vid_directory : TYPE str
        DESCRIPTION. location of videos to chekc frames of 
    frame_list : TYPE array
        DESCRIPTION. either A) array of frame numbers manually inputted (set manual to TRUE)
                    or B) list of either frame numbers times of frames in msec -that is part of dlc_data_struc eg trial_start_times_MED
    frame_range : TYPE array of two elements
        DESCRIPTION. first elemment no of frames before and after event to chek eg +- 30
                        second elemenet: step value of range ef g 
    all_data : TYPE class - dlc data struc
        DESCRIPTION. used to get total number of frames in video from .dlc_data length
    method : TYPE str
        DESCRIPTION. either 'frames' for frame number or 'times_msec' for frame time in msecs
    manual : TYPE bool
        DESCRIPTION. set True to pass freely set list of frame numbers or False to pass attr of dlc_data_struc

    Returns
    -------
    None.

    """
    for tag in tag_list:
        files = os.listdir(vid_directory)
        video_name = [x for x in files if tag in x and '.mp4' in x or '.avi' in x ][0]
        os.chdir(vid_directory)
        cap = cv2.VideoCapture(video_name)
        if manual:
            events = frame_list
        else:
            events =  getattr(all_data[tag],frame_list) 
        for event in events:
            for f in range(int(event-frame_range[0]),int(event+frame_range[1]),frame_range[2]):
                if method == 'frame':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                elif method == 'time_msec':
                    cap.set(cv2.CAP_PROP_POS_MSEC, f)
                res, frame = cap.read()
           
                #Display the resulting frame
                cv2.imshow(tag+' frame '+ str(event),frame)
                
                #Set waitKey 
                cv2.waitKey()
            
                cv2.destroyAllWindows()
        cap.release()
        

def collect_traj_by_trialtype(trials_to_analyse,sessions_to_analyse,all_data,mismatched_files,scaled =True):
    '''
    
    creates a dictions for each trial type of interest, for each session of interest, where cell contains the tracking data for each individual trial
    under those coniditons (form all animals)
    
    eg traj_by_trial[trialtype][session]{each trial of that type: data}
    
    
    also returns dicts for each trial type and dose with the percantge of nan - non tracked points - for head and nose labels
    Parameters
    ----------
    trials_to_analyse : TYPE list
        DESCRIPTION. list detailing trials of interest
    sessions_to_analyse : TYPE list
        DESCRIPTION. details sessions of interest
    all_data : TYPE dlc data struc
        DESCRIPTION. used to get all tags

    Returns
    -------
    all_traj_by_trial: TYPE dict
        DESCRIPTION. contians by-trial tracking data sorted by trial type and session 
    all_nose_nan_perc_trials: TYPE dict
         DESCRIPTION. contians trial nan % for nose and head in df label sorted by trial type and session 
     
    '''
    str_for_querys = {'go1_succ': 'success go single',
                      'go1_rtex': '\ufeffmax response time exceeded go single',
                      'go1_wronglp':'incorrect go single',
                      'go2_succ':  'success go double',
                      'go2_rtex': '\ufeffmax response time exceeded go double',
                      'go2_wronglp': 'incorrect go double'}
    
    all_traj_by_trial = {}
    nan_perc_diag = {}
    if scaled:
        tracking = 'track_by_trial_scaled'
    else:
        tracking = 'track_by_trial_norm'
        
    for tt in trials_to_analyse:
        traj_by_trial = {}
        nan_perc_diag_session= {}
        
        for session in sessions_to_analyse:
            sel_trials = {}
            nan_perc_diag_trials  =pd.DataFrame()
            session_tags = [x for x in list(all_data.keys()) if session in x ]
            for tag in session_tags:
                print(tag)
                if tag not in mismatched_files:
                    outcome_key = str_for_querys[tt]
                    trial_numbers = all_data[tag].trial_data.query('`outcome` == @outcome_key')
                    tracking_data = getattr(all_data[tag],tracking)
                    for trial in trial_numbers.index:
                        #print(trial)
                        if trial in tracking_data.keys(): #account for some videos cut short so not covering trial 
                            if all_data[tag].double_side == 1: 
                                sel_trials[tag + str(trial)] =  tracking_data[trial]
                                
                                nose_perc = tracking_data[trial]['nose'].isnull().sum()/len(tracking_data[trial]['nose'])
                                head_perc= tracking_data[trial]['head'].isnull().sum()/len(tracking_data[trial]['head'])
                                nan_perc_diag_trials.loc[tag + str(trial),'nose']  = nose_perc.iloc[0]
                                nan_perc_diag_trials.loc[tag + str(trial),'head'] =head_perc.iloc[0]
                                 
                                if tracking_data[trial]['head'].isnull().values.any():
                                    print('nans here!' + ' ' + tag)
                            elif all_data[tag].double_side == 2:
                                flipped_trial = pd.DataFrame().reindex_like( tracking_data[trial])
                                flipped_trial[('nose','x')] = tracking_data[trial]['nose'].x *-1
                                flipped_trial[('head','x')] = tracking_data[trial]['head'].x *-1
                                for val in ['y','likelihood']:
                                    for part in ['nose','head']:
                                        
                                        flipped_trial[(part,val)] = tracking_data[trial][part][val]
                                other_parts =  list(all_data[tag].bodyparts)
                                other_parts.remove('head')
                                other_parts.remove('nose')
                                for o_part in other_parts:
                                    flipped_trial[o_part] = tracking_data[trial][o_part]
                                sel_trials[tag + str(trial)] = flipped_trial
                                
                                nose_perc = flipped_trial['nose'].isnull().sum()/len(flipped_trial['nose'])
                                head_perc = flipped_trial['head'].isnull().sum()/len(flipped_trial['head'])
                                nan_perc_diag_trials.loc[tag + str(trial),'nose'] = nose_perc.iloc[0]
                                nan_perc_diag_trials.loc[tag + str(trial),'head'] = head_perc.iloc[0]
                                if flipped_trial['head'].isnull().values.any():
                                    print('nans here!' + ' ' + tag)
    
            traj_by_trial[session] = sel_trials
            nan_perc_diag_session[session] = nan_perc_diag_trials
        all_traj_by_trial[tt] = traj_by_trial
        nan_perc_diag[tt] =  nan_perc_diag_session
        
    return all_traj_by_trial, nan_perc_diag
                  
def get_avg_norm_medians(all_data):
    '''
    

    Parameters
    ----------
    all_data : TYPE dlc_data struc
        DESCRIPTION.

    Returns
    -------
    avg_all_norm_medians : df
        DESCRIPTION. df with normalised medians of box features averaged across indiviual animal+dose sessions

    '''
    all_norm_medians = pd.DataFrame()
    for tag in all_data.keys():
        for k in all_data[tag].box_norm_medians:
            all_norm_medians.loc[tag,k] = all_data[tag].box_norm_medians[k]
            
    avg_all_norm_medians = all_norm_medians.mean()
    return avg_all_norm_medians


def plot_trajectories(trials_to_plot, sessions_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,subj_to_plot, by_subject =True) :
    '''
    
    plots all trajections grouped by trial type and session 
    
    each trial type generates a plot with subplots for each dose 

    Parameters
    ----------
    trials_to_plot : TYPE list
        DESCRIPTION. trial types to plot out 
    sessions_to_plot : TYPE list
        DESCRIPTION. sessions to plot
    traj_part : TYPE str
        DESCRIPTION. dlc label to plot traj by eg head or nsoe 
    all_traj_by_trial : TYPE dict
        DESCRIPTION. dict of all trajectories
    avg_all_norm_medians : TYPE df
        DESCRIPTION. df of medians of box features 

    Returns
    -------
    None.

    '''
    import dlc_functions as dlc_func
    import matplotlib.patches as mpatches
    traj_colors = ['grey','limegreen', 'darkgreen']
    indv_colors = ['darkred','orangered','goldenrod','olive','lawngreen','forestgreen','mediumaquamarine','cyan','dodgerblue','navy','rebeccapurple','plum','saddlebrown','gray','black']
    all_patches ={}
    for tt in range(0,len(trials_to_plot)):
        if len(sessions_to_plot) > 1:
            
            fig, axs = plt.subplots(1,len(sessions_to_plot),figsize = (16,4),sharex = True, sharey = True)
        else:
            fig, axs = plt.subplots(1,len(sessions_to_plot)+1,figsize = (16,4),sharex = True, sharey = True)
    
        axs =axs.ravel()
        trial_type_data= all_traj_by_trial[trials_to_plot[tt]]
        for s in range(0,len(sessions_to_plot)):
            print(s)
            session_data = trial_type_data[sessions_to_plot[s]]
            print(sessions_to_plot[s])
            if by_subject:
                all_patches[s] = []
                sub_counter =0
                indv_subjs = ['rat' + x.split('rat',1)[1][0:2] for x in session_data.keys()]  
                print(indv_subjs)
                if subj_to_plot == 'all':
                    indv_subjs_filt = indv_subjs
                else:
                    indv_subjs_filt = [ x for x in indv_subjs if x in subj_to_plot]
                    
                indv_subjs_filt_set = set(indv_subjs_filt)
                for subj in indv_subjs_filt_set:
                    subj_data = [x for x in session_data.keys() if subj in x ]
                    for trial in subj_data:
                        data_x = session_data[trial][traj_part].x
                        data_y = session_data[trial][traj_part].y
                        axs[s].plot(data_x,data_y,indv_colors[sub_counter], alpha=0.3)
                    p =mpatches.Patch(color =indv_colors[sub_counter], label =subj, alpha = 0.3)
                    all_patches[s].append(p)
                    sub_counter +=1
            else:    
                for trial in session_data.keys():
                    data_x = session_data[trial][traj_part].x
                    data_y = session_data[trial][traj_part].y
                    axs[s].plot(data_x,data_y,traj_colors[s], alpha=0.3)
            axs[s].set_title(sessions_to_plot[s],fontsize =20)
            dlc_func.plotbox(axs[s], avg_all_norm_medians)
            plt.suptitle(trials_to_plot[tt],fontsize =26, y=1.1)
            if by_subject:
                axs[s].legend(handles = all_patches[s],loc ='best',fontsize=7, bbox_to_anchor =(1.2,1))
            plt.show()

def plot_indv_trajectories(trials_to_plot,sessions_to_plot,animals_to_plot,
                           traj_part,all_traj_by_trial,avg_all_norm_medians,
                           num_traj,all_data,plot_by_traj,
                           all_occ_ords = None, print_occ_ord = False):
    traj_colors = ['grey','limegreen', 'darkgreen']
    for tt in range(0,len(trials_to_plot)):
        trial_type_data= all_traj_by_trial[trials_to_plot[tt]]
        for s in range(0,len(sessions_to_plot)):
            session_data = trial_type_data[sessions_to_plot[s]]
            for animal in animals_to_plot:
                animal_tag = 'rat'+animal
                session_trials = [x for x in list(session_data.keys()) if animal_tag in x]
                counter = 0
                tag = animal_tag + '_'+ sessions_to_plot[s]
                if not plot_by_traj:
                    fig,ax = plt.subplots(1,1)
                    plt.title(animal_tag +' ' + sessions_to_plot[s] )

                for trial in  session_trials:
                    counter +=1
                    if counter < num_traj+1:
                        if plot_by_traj:
                            fig,ax = plt.subplots(1,1)
                            if print_occ_ord ==True:     
                                
                                plt.title(all_occ_ords[trials_to_plot[tt]][sessions_to_plot[s]][trial])
                            else:
                                
                                plt.title(trial)
                            
                        data_x = session_data[trial][traj_part].x
                        data_y = session_data[trial][traj_part].y
                        plt.plot(data_x,data_y,traj_colors[s], alpha=0.3)  
                        plotbox(ax, all_data[tag].box_norm_medians)
                        
                            
                       

def pdf_heatmaps(trials_to_plot,sessions_to_plot,all_traj_by_trial,traj_part,n_bins):
    
    all_pdfs = {}
    for tt in trials_to_plot:
        trial_type_data= all_traj_by_trial[tt]
        session_pdfs = {}
        for s in sessions_to_plot:
            session_data = trial_type_data[s]
            trial_pdf ={}
            for trial in session_data.keys():
                trial_pdf[trial],_,_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                                   bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density =True)       
            session_pdfs[s] = trial_pdf
        all_pdfs[tt] = session_pdfs
        #using a restricted map where area behaind the magazine is excluded


    for tt in all_pdfs.keys():
        for s in all_pdfs[tt].keys():                                    
            #x =np.mean([trial_pdf[i] for i in trial_numbers])
            pdf_arr = np.array(list(all_pdfs[tt][s].values()))
            pdf_mean = pdf_arr.mean(axis=0)
            pdf_log_mean = np.log(pdf_mean)
            f, ax = plt.subplots(1,1)
            ax.imshow(pdf_log_mean, cmap='hot')#,ax=ax)
            ax.set_title(tt + ' ' + s)

def calc_traj_length(trials_to_plot,sessions_to_plot,all_traj_by_trial,avg_all_norm_medians,traj_part):

    median_poke = np.array([float(avg_all_norm_medians['poke_x']),float(avg_all_norm_medians['poke_y'])])
    median_llever = np.array([float(avg_all_norm_medians['l_lever_x']),float(avg_all_norm_medians['l_lever_y'])])

    lever_distance =  np.linalg.norm(median_llever-median_poke) # find eucliden distance 
    traj_distance = pd.DataFrame()
    mean_traj_distance = pd.DataFrame()

    for tt in trials_to_plot:
        trial_type_data= all_traj_by_trial[tt]
        for s in sessions_to_plot:
            session_data = trial_type_data[s]
            for trial in session_data.keys():
        #calc distance using pythag
                distance = np.sqrt(session_data[trial][traj_part].x.diff()**2 +session_data[trial][traj_part].y.diff()**2)
                traj_distance.loc[trial,'total_distance'] = distance.sum()           
                traj_distance.loc[trial,'norm_total_distance'] = distance.sum()/lever_distance
                traj_distance.loc[trial,'dose'] = s
                traj_distance.loc[trial,'trial_type'] = tt
                traj_distance.loc[trial,'animal'] = trial[3:5]
            mean_norm_total_distance = traj_distance.norm_total_distance.mean()
            mean_traj_distance.loc[tt,s] = mean_norm_total_distance
        
    mean_traj_distance.plot(kind='bar')
    return traj_distance, mean_traj_distance


def clean_and_interpolate_dlc_data(criterion, all_data):
    """
    This function interpolates across frames where DLC is unconfident about bodypart position

    Parameters
    ----------
    criterion : confidence level at which to interpolate across 
    
    all_data : dict of data_strucs
    
    
    Returns
    -------
    all_data : returns dict with each struc now containing "dlc_data"
    confidence_prop: table with proportion frames where confidence is above criteron for each body part, for each video 
    """
    confidence_prop = pd.DataFrame()
    for tag in all_data.keys():
        all_data[tag].dlc_data = pd.DataFrame().reindex_like(all_data[tag].dlc_data_preproc)
        
        
        for part in all_data[tag].bodyparts:
            
            conf_idx_data = all_data[tag].dlc_data_preproc[part].query('likelihood > @criterion')
            confidence_prop.loc[tag,part] = len(conf_idx_data)/len(all_data[tag].dlc_data_preproc[part])

            df_copy = pd.DataFrame().reindex_like(all_data[tag].dlc_data_preproc[part])
            
            df_copy.iloc[conf_idx_data.index,df_copy.columns.get_loc('x')] = conf_idx_data.x
            df_copy.iloc[conf_idx_data.index,df_copy.columns.get_loc('y')] = conf_idx_data.y
            
            # interpolate between low confidence values - if there are nans at the start of the column - then will not extrapolate backwars
            # instead just fill with the first non NAN (eg conf above criterion) value
            
            df_copy.x = df_copy.x.interpolate(limit_direction = 'both', limit = 10)
            df_copy.y = df_copy.y.interpolate(limit_direction = 'both', limit = 10)
            
            df_copy.likelihood = all_data[tag].dlc_data_preproc[part].likelihood
            all_data[tag].dlc_data[part] = df_copy
    
    return all_data, confidence_prop

