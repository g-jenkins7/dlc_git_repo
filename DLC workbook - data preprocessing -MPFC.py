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
collect_beh_data = 'n'
process_all_videos = 'n'
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



if collect_beh_data == 'y':
    mpfc_all_data = {}

    for exp in ['IL','PL','MO']:
        session_file_path = data_path + 'allSessions_allSubjects_mPFC_' + exp + '.p'
        all_beh_data = pickle.load(open (session_file_path , 'rb')) 
        all_beh_data['sal'] =all_beh_data.pop('SAL') 
        all_beh_data['bm'] =all_beh_data.pop('BMI') 
        subject_list = list(all_beh_data['sal'].keys())
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
    mpfc_all_data,valid_points  = dlc_func.interpolate_dlc_data(mpfc_all_data, 2)

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

#%%

mpfc_all_data, distances = dlc_func.normalise_and_scale(mpfc_all_data,all_frame_times,all_mismatched_files)

#%%
mpfc_all_data = dlc_func.track_all(mpfc_all_data, all_mismatched_files,distances, restrict_traj = True)
#%%
for key in mpfc_all_data.keys():
    print(key)
    print(mpfc_all_data[key].dlc_data_norm.shape)
#%%
succ_extra_time = 0.75
frame_time_in_sec = 0.04#
extra_time = np.floor(succ_extra_time/frame_time_in_sec)

extra_time = np.floor(succ_extra_time/frame_time_in_sec)
track_trials(mpfc_all_data['rat01_08_sal'], mpfc_all_data['rat01_08_sal'].dlc_data_norm,extra_time)

#%%


D = test['rat02_13_bmi'].dlc_data_norm
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
    first_mag_entry = in_roi[in_roi.index > succ_time[0]].iloc[0].name
    succ_tracking = data[D.trial_start_frames.iloc[trial_no]: first_mag_entry + float(extra_time)]   
    
    
#%%
for tag in all_data.keys():
    if tag not in excluded_files:
        print(tag)
# get median of box features across trials 
        frame_time_in_sec = 0.04# = float(all_frame_times[tag][1])
        print(frame_time_in_sec)
        extra_time = np.floor(succ_extra_time/frame_time_in_sec) #N frames after succ to track succ trials 
        
        all_data[tag].box_medians = get_medians(all_data[tag]) # get medians for box features 

        all_data[tag].dlc_data_norm = normalise_dlc_data(all_data[tag]) # subtract poke median to normalise data

        all_data[tag].box_norm_medians = get_medians(all_data[tag], norm = True) #get medians of normalised data 

    all_avg_norm_medians = dlc_func.get_avg_norm_medians(all_data) # avg medians for plotting 
    
    distances = dlc_func.get_distances(all_data,all_avg_norm_medians)
    
    
    
#%%   
    #find brightness peaks
    # plot_peaks = 'err_only' # for find brightness peaks function   
    # all_peaks, all_onsets, mismatch_list_sal = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,
    #                                                                               mpfc_all_data,plot_peaks,
    #                                                                               subject_list, ['bmi'])
#%%
    #align data                                                                              
    period = 5 # time in s after trial iniation to track across
    mpfc_all_data, all_mismatched_files = dlc_func.get_trial_frames(mpfc_all_data,all_onsets,all_frame_times,period)
    all_mismatched_files = all_mismatched_files + unstable_frame_rate_files
    
    
    #uncomment to check MED-pc error alignment with brightness peak onsets detected in the video 
    # check_err =  True
    # all_peaks = dlc_func.find_brightness_peaks_dspk(all_frame_times,all_avg_brightness,
    #                                                                               mpfc_all_data,plot_peaks,
    #                                                                               subject_list, ['sal'],check_err)#,'bmi'])
    
    
    #cut data into trials
    restricted_traj=True #restricts points outside the box limits eg rearing for head trajectories to within confines of the box floor
    mpfc_all_data = dlc_func.normalise_and_track(mpfc_all_data,'head',0.75,all_frame_times,all_mismatched_files,restricted_traj)

    

    pickle.dump(mpfc_all_data, open(data_path + 'all_data_' + 
                               experiment +
                               '_preproc.p', 'wb')) 
    
    pickle.dump(all_mismatched_files, open(data_path + 'mismatched_files' + 
                               experiment +
                               '.p', 'wb')) 

# else:
    
#     all_data = pickle.load(open (data_path + 'all_data_' + 
#                                experiment +
#                                '_preproc.p', 'rb'))

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
