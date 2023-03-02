# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:00:59 2022

author: George
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
import dlc_functions as dlc_func
from scipy import interpolate


#10mg = dog
#1mg = fox
#veh = cat

#rename files in pwoershell 

# ls |Rename-Item -NewName {$_.name -replace "cat","veh"}





#%%
collect_beh_data = 'y'
process_all_videos = 'y'
preprocess_data = 'y'

#%% COLLECTING BEHAVIOURAL DATA 
#set subject and session
session_list = ['sal','bm']
experiment = 'mpfc'

#behavioural file informaiton 
data_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - MPFC/'
all_trials_path = data_path + 'trials_all_mPFC.p'  
all_trial_data = pickle.load(open (all_trials_path , 'rb')) 
#all_beh_data['veh'] =all_beh_data.pop('VEH') 
#all_trial_data[0].session = all_trial_data[0].session.replace({'VEH':'veh'})
#dlc file informaiton 
dlc_file_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - MPFC/DLC_FILES/gongo inactivations/'
subject_list = list(all_trial_data.animal.unique())

all_trial_data.loc[:,'session'] = ['bm'
                                    if x == 'BMI'
                                    else 'sal'
  
                                    for x in all_trial_data.session]

if collect_beh_data == 'y':
    mpfc_all_data = {}

    for exp in ['IL','PL','MO']:
        session_file_path = data_path + 'allSessions_allSubjects_mPFC_' + exp + '.p'
        all_beh_data = pickle.load(open (session_file_path , 'rb')) 
        all_beh_data['sal'] =all_beh_data.pop('SAL') 
        all_beh_data['bm'] =all_beh_data.pop('BMI') 
        subject_list = list( all_beh_data['sal'].keys())
        all_data = dlc_func.get_beh_dlc_data(session_list,
                                             subject_list,
                                             all_beh_data,
                                             all_trial_data,
                                             dlc_file_path) 
        pickle.dump(all_data, open(data_path + 'all_data_' + 
                                   experiment + exp + 
                                   '.p', 'wb')) 
        mpfc_all_data.update(all_data)
        print(mpfc_all_data.keys())
        pickle.dump(mpfc_all_data, open(data_path + 'all_data_' + 
                                   experiment + 
                                   '.p', 'wb')) 
elif preprocess_data == 'y':
    mpfc_all_data = pickle.load(open (data_path + 'all_data_' + 
                               experiment +
                               '.p', 'rb'))

#%%
for subject in subject_list:
    print(subject)
    print(all_trial_data.query('animal == @subject').shape)
    
#%% VIDEO PROCESSING - ffmpeg timeframe exdtraction and avg brightness data


if process_all_videos == 'y':
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - MPFC/DLC_VIDEOS'
    old_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc_mpfc.p', 'rb')) 
    old_avg_brightness = pickle.load( open(dlc_file_path + '/all_avg_brightness_dlc_mpfc.p', 'rb')) 
    all_frame_times,   all_avg_brightness = dlc_func.process_videos(vid_directory,
                                                                     experiment,                                                                    
                                                                     old_frame_times,
                                                                     old_avg_brightness,
                                                                     update = True)
    
    os.chdir(dlc_file_path)    
    pickle.dump(all_frame_times, open('all_frametimes_dlc_mpfc.p', 'wb'))
    pickle.dump(all_avg_brightness, open('all_avg_brightness_dlc_mpfc.p', 'wb'))
    
else: 
    all_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc_mpfc.p', 'rb')) 
    all_avg_brightness = pickle.load( open(dlc_file_path + '/all_avg_brightness_dlc_mpfc.p', 'rb')) 
    frame_rate_test = pd.DataFrame()
    for key in all_frame_times.keys():
        frames = [float(x) for x in all_frame_times[key]]
        if frames:
            frame_rate_test.loc[key,'avg'] = np.mean(np.diff(frames))
            frame_rate_test.loc[key,'max'] = np.max(np.diff(frames))
            frame_rate_test.loc[key,'min'] = np.min(np.diff(frames))
            frame_rate_test.loc[key,'std'] = np.std(np.diff(frames))
        else:
            frame_rate_test.loc[key,'avg'] = np.nan
            frame_rate_test.loc[key,'max'] = np.nan
            frame_rate_test.loc[key,'min'] = np.nan
            frame_rate_test.loc[key,'std'] = np.nan

    print(frame_rate_test.loc[frame_rate_test['avg'] != 0.04])
    unstable_frame_rate_files = list(frame_rate_test.loc[frame_rate_test['avg']  !=  0.04].index)
    
#%% REMOVING COHORT 2 FILES 


# all_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc_mpfc_pre_file_rename.p', 'rb')) 
# all_avg_brightness = pickle.load( open(dlc_file_path + '/all_avg_brightness_dlc_mpfc_pre_file_rename.p', 'rb')) 
# cohort2 = [x for x in list(all_frame_times.keys()) if 'rat02' in x]
# for f in cohort2:
#     del all_frame_times[f]
#     del all_avg_brightness[f]
# pickle.dump(all_frame_times, open(dlc_file_path+ '/all_frametimes_dlc_mpfc.p', 'wb'))
# pickle.dump(all_avg_brightness, open(dlc_file_path+ '/all_avg_brightness_dlc_mpfc.p', 'wb'))

#%% interpolate DLC data, find brightness peaks from the video - align to MED-file errors and convert MED events into frames

excluded_vids =['rat02_05_bm',
                'rat02_04_sal',
                'rat02_10_sal',
                'rat02_04_bm',
                'rat02_10_bm',
                'rat02_24_bm'] # these videos not analysed yet 
for k in excluded_vids:
    if k in mpfc_all_data.keys():
        mpfc_all_data.pop(k)
    if k in all_frame_times.keys():
        all_frame_times.pop(k)
    if k in all_avg_brightness.keys():
        all_avg_brightness.pop(k)

mpfc_all_data = {(k +'i' if 'bm' in k else k): v for (k,v) in mpfc_all_data.items()}

for tag in mpfc_all_data.keys():
    setattr(mpfc_all_data[tag], 'tag', tag)
            
            

#%%
for tag in mpfc_all_data.keys():
    setattr(mpfc_all_data[tag], 'error_alignment_setting', {'start_frame': 400 , #n frames from start of session to skip - sometimes door is open
                                                            'end_frame': -1, # last frame to take - soem files have messy endings
                                                            'spike_threshold': 5, # remove high spikes in brightness
                                                            'rebound_threshold': -5, #remove rebounds 
                                                            'peak_scale_factor':  1.53, #SNR to scale by for peak detecetiond
                                                            'find_peaks_distance':  110, #distance measure for scipy find peaks functions
                                                            'upshift': 0, #boost every birghtness by upshift - some files end up wiht negative vals after filtering/normalising
                                                            'brightness_sample_offset': 0}) # where to take 'average' brightness sample from - default is frames 10,000:20,000
    
mpfc_all_data['rat02_13_bmi'].error_alignment_setting['find_peaks_distance'] = 90
mpfc_all_data['rat02_15_bmi'].error_alignment_setting['end_frame'] = 95000
mpfc_all_data['rat02_15_bmi'].error_alignment_setting['upshift'] = 20
mpfc_all_data['rat02_14_sal'].error_alignment_setting['find_peaks_distance'] = 70
#mpfc_all_data['rat02_05_bmi'].error_alignment_setting['find_peaks_distance'] = 70
mpfc_all_data['rat02_20_bmi'].error_alignment_setting['brightness_sample_offset'] = 20000
mpfc_all_data['rat02_16_bmi'].error_alignment_setting['brightness_sample_offset'] = -5000
mpfc_all_data['rat02_11_bmi'].error_alignment_setting['start_frame'] = 3000
mpfc_all_data['rat02_11_bmi'].error_alignment_setting['brightness_sample_offset'] = 20000
mpfc_all_data['rat01_07_sal'].error_alignment_setting['find_peaks_distance'] = 70
mpfc_all_data['rat02_07_sal'].error_alignment_setting['find_peaks_distance'] = 70
mpfc_all_data['rat01_18_bmi'].error_alignment_setting['brightness_sample_offset'] = 40000

if preprocess_data == 'y':
    
    
    
    #interpolate
    mpfc_all_data,valid_points  = dlc_func.interpolate_dlc_data(mpfc_all_data, 2, lik_threshold =0.9, jump_threshold=20, min_len=10)

# mismatch_list = []
# plot_peaks = 'y'
# f_onsets, peaks, _ = get_peaks(all_frame_times,all_avg_brightness,mpfc_all_data, plot_peaks, 'rat01_11_sal', mismatch_list, check_err = True)


    plot_peaks = 'n'
    all_peaks, all_onsets,_ =  dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,
                               mpfc_all_data,plot_peaks,
                               subject_list, ['sal','bmi'])

    all_mismatched_files = []
    period = 5 # time in s after trial iniation to track across
    mpfc_all_data, all_mismatched_files = dlc_func.get_trial_frames(mpfc_all_data,all_onsets,all_frame_times,period)
    #all_mismatched_files = all_mismatched_files + unstable_frame_rate_files


    mpfc_all_data, distances = dlc_func.normalise_and_scale(mpfc_all_data,all_frame_times,all_mismatched_files)


    mpfc_all_data = dlc_func.track_all(mpfc_all_data, all_mismatched_files,distances, restrict_traj = True)


    

    pickle.dump(mpfc_all_data, open(data_path + 'all_data_' + 
                               experiment +
                               '_preproc.p', 'wb')) 
    
    pickle.dump(all_mismatched_files, open(data_path + 'mismatched_files_' + 
                               experiment +
                               '_preproc.p', 'wb')) 

else:
    
    mpfc_all_data = pickle.load(open (data_path + 'all_data_' + 
                                experiment +
                                '_preproc_jt20_ml10.p', 'rb'))
    all_mismatched_files = pickle.load(open (data_path + 'mismatched_files_' + 
                                experiment +
                                '_preproc.p', 'rb'))
#%%




#%%  chopping data into trials
#avg_all_norm_medians = dlc_func.get_avg_norm_medians(all_data)
#distances = dlc_func.get_distances(all_data,avg_all_norm_medians)



#%% frame checker - examine whats going on in a section of the vid to check it aligns with trajecotires
#FIX frame chekcer ???

check_frames  = 'no'
start_frame = 0
end_frame = 200
frame_step = 5
frame_range = [start_frame,end_frame,frame_step]

manual = True
if check_frames == 'yes':
    tag_list = ['rat06_10mg']
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analys test videos'
    dlc_func.frame_checker(tag_list,vid_directory,[38910],frame_range,all_data,'frame',manual)
     
#%% 
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~


#%%

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#NOTES ON DLC ANALYSIS PROCESS
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
# get beh dlc data 
# to get trial succ times - get list to succ MED timestamps, find closest trial start time that preceeds the succ ttl and assign succ time to that trial


# processing video 
# using ffmpeg show info to get frame times
# using openCV2 to get average brightness

# interpolating data
# liklihood threshold set to 0.9 -  anything below this from DLC discarded 
# jump threshold set to 10 - differences in coords > 10*sd dev are discarded (stdev estimated from percentiles)
# isolated points, any run of less than 5 points surrounded by NaNs are discarded 


#finding error peaks
# normalise the birghtness by mean subtraction 
# remove spikes and rebounds by setting indexs where change in brihgtness is 5 < or < -5 to nan and interpolating 
# setting peak prominance to 1.53 x the average brightness from a middle portion of the session(frames 10,000 to 20,000)
# using scipy peak to find peaks  and filter them by prominences
# finding the sharpest change in d_brightness in the 250 idxs before peak to get onset of brightness and take that as error onset 

#converting med times into frames
# taking average difference in frame times and converting to seconds
# finding difference in s between frame times from med and brighness peaks from video
# discaridng files with greater than 500ms range of differences between med errors and video peaks 
# taking the average of the differences between med errors and video peaks as the difference in time between video starting and med session starting 
# adding the video start time to med times and dividing by frame times in second to get frames of beh events 
# time taken from trial start to end of tracking set to 7s (for failed trials)

#chopping data into trials 
# get medians of the box features - subtracting individual files poke median from the tracking data to normalise
# restricting trajectorys to within the floor of the box (in y domain) by setting anything greater than the y-level of the food mag to zeor 
# scaling trajectories in the x domain by relative lever distances - at the moment leaving y domain untouched
# chopping failed trials between trial start and trial start +7s 
# chopping successful trials between trials start and 0.75s after food mag roi entry following successful trial completion for succesful trials

# collecting trajectory by trial type 
# flipping trials if animals double side is 2 so that go small is always on the left and go large always on the right
