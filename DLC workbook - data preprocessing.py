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
process_videos = 'n'
preprocess_data = 'y'

#%% COLLECTING BEHAVIOURAL DATA 
#set subject and session
subject_list = ['05','06','07','08','09','11','12','13','14','19','20','21','22','23','24']
session_list = ['veh','1mg','10mg']
experiment = 'ecb'

#behavioural file informaiton 
data_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/'
session_file_path = data_path +'allSessions_allSubjects_ecb.p'  
all_trials_path = data_path + 'all_trials_ecb.p'  
all_beh_data = pickle.load(open (session_file_path , 'rb')) 
all_trial_data = pickle.load(open (all_trials_path , 'rb')) 
all_beh_data['veh'] =all_beh_data.pop('VEH') 
all_trial_data[0].session = all_trial_data[0].session.replace({'VEH':'veh'})
#dlc file informaiton 
dlc_file_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analysis test dlc files'


if collect_beh_data == 'y':
    all_data = dlc_func.get_beh_dlc_data(session_list,
                                         subject_list,
                                         all_beh_data,
                                         all_trial_data,
                                         dlc_file_path) 
    pickle.dump(all_data, open(data_path + 'all_data_' + 
                               experiment +
                               '.p', 'wb')) 
elif preprocess_data == 'y':
    all_data = pickle.load(open (data_path + 'all_data_' + 
                               experiment +
                               '.p', 'rb'))
#%% VIDEO PROCESSING - ffmpeg timeframe exdtraction and avg brightness data


if process_videos == 'y':
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analys test videos'
    all_frame_times,    all_avg_brightness = dlc_func.process_videos(vid_directory)
    
    os.chdir(dlc_file_path)
    pickle.dump(all_frame_times, open('all_frametimes_dlc.p', 'wb'))
    pickle.dump(all_avg_brightness, open('all_avg_brightness_dlc.p', 'wb'))
else: 
    all_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc.p', 'rb')) 
    all_avg_brightness = pickle.load( open(dlc_file_path + '/all_avg_brightness_dlc.p', 'rb')) 
    frame_rate_test = pd.DataFrame()
    for key in all_frame_times.keys():
        frames = [float(x) for x in all_frame_times[key]]
        frame_rate_test.loc[key,'avg'] = np.mean(np.diff(frames))
        frame_rate_test.loc[key,'max'] = np.max(np.diff(frames))
        frame_rate_test.loc[key,'min'] = np.min(np.diff(frames))
        frame_rate_test.loc[key,'std'] = np.std(np.diff(frames))

    print(frame_rate_test.loc[frame_rate_test['avg'] > 0.04])
    unstable_frame_rate_files = list(frame_rate_test.loc[frame_rate_test['avg'] > 0.04].index)
    
                    


#%% interpolate DLC data, find brightness peaks from the video - align to MED-file errors and convert MED events into frames

if preprocess_data == 'y':
    
    #interpolate
    all_data,valid_points  = dlc_func.interpolate_dlc_data(all_data, 2)
   
    #find brightness peaks
    plot_peaks = 'n' # for find brightness peaks function   
    all_peaks, all_onsets, mismatch_list_10 = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,
                                                                                  all_data,plot_peaks,
                                                                                  subject_list,['veh','1mg','10mg'])

    #align data                                                                              
    period = 5 # time in s after trial iniation to track across
    all_data, all_mismatched_files = dlc_func.get_trial_frames(all_data,all_onsets,all_frame_times,period)
    all_mismatched_files = all_mismatched_files + unstable_frame_rate_files
    
    
    #uncomment to check MED-pc error alignment with brightness peak onsets detected in the video 
    #check_err =  True
    #all_peaks = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,all_data,plot_peaks,subject_list,['veh'],check_err)
    
    
    #cut data into trials
    restricted_traj=True #restricts points outside the box limits eg rearing for head trajectories to within confines of the box floor
    all_data = dlc_func.normalise_and_track(all_data,'head',0.75,all_frame_times,all_mismatched_files,restricted_traj)

    

    pickle.dump(all_data, open(data_path + 'all_data_' + 
                               experiment +
                               '_preproc.p', 'wb')) 
    
    pickle.dump(all_mismatched_files, open(data_path + 'mismatched_files' + 
                               experiment +
                               '.p', 'wb')) 

else:
    
    all_data = pickle.load(open (data_path + 'all_data_' + 
                               experiment +
                               '_preproc.p', 'rb'))

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

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

trials_to_analyse = ['go1_succ','go1_rtex']#, 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_analyse = ['veh','1mg','10mg']
 
all_traj_by_trial, all_nose_nan_perc_trials, all_head_nan_perc_trials = dlc_func.collect_traj_by_trialtype(trials_to_analyse,sessions_to_analyse,all_data,all_mismatched_files,scaled = False)
               
scaled_all_traj_by_trial, all_nose_nan_perc_trials, all_head_nan_perc_trials = dlc_func.collect_traj_by_trialtype(trials_to_analyse,sessions_to_analyse,all_data,all_mismatched_files,scaled = True)

avg_all_norm_medians = dlc_func.get_avg_norm_medians(all_data)


#%%
trials_to_plot = ['go1_succ','go1_rtex']#, 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_plot = ['veh','1mg','10mg']#['veh','1mg','10mg']
traj_part = 'head'

by_subject = True
subj_to_plot =[ 'rat10','rat12','rat24','rat05']
dlc_func.plot_trajectories(trials_to_plot, sessions_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,subj_to_plot,by_subject)


#%% individual trial plotting

trials_to_plot = ['go1_succ']# ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
sessions_to_plot = ['veh']#,'1mg','10mg']
num_traj = 10
plot_by_traj =False
animals_to_plot = ['06','13']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,all_data,plot_by_traj)



 


#%%

#%%

#%%%

#%% PDF - heat maaps
n_bins = 18

all_pdfs = {}
for tt in trials_to_plot:
    trial_type_data= all_traj_by_trial[tt]
    session_pdfs = {}
    for s in sessions_to_plot:
        session_data = trial_type_data[s]
        trial_pdf ={}
        for trial in session_data.keys():
            trial_pdf[trial],_,y_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
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

#loggingg the values

#%%

trial_pdf ={}
for trial in session_data.keys():
    trial_pdf[trial],_,y_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                       bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density =True) 

















#%% calculating trajectory length - normalise by distance between levers

# CHECK need to crop traj so they end at the food magazine? maybe for correct trials - after succ trigger = go to food mag
# for incorrect trials  - after 5 sec timeout? 
traj_distance, mean_traj_distance = dlc_func.calc_traj_length(trials_to_plot,sessions_to_plot,all_traj_by_trial,avg_all_norm_medians,traj_part)



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
