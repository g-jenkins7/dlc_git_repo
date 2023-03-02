# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:52:23 2023

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
import dlc_functions as dlc_func
from scipy import interpolate
import scipy.stats as st
import pingouin as pg 



#%%
experiment = 'mpfc'


data_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - MPFC/'


mpfc_all_data = pickle.load(open (data_path + 'all_data_' + 
                            experiment +
                            '_preproc.p', 'rb'))
all_mismatched_files = pickle.load(open (data_path + 'mismatched_files_' + 
                             experiment +
                             '_preproc.p', 'rb'))
#%%


extract_traj = True
plot_traj = True
plot_heat_maps = False

all_mismatched_files = all_mismatched_files + ['rat01_14_sal']
#%%

if extract_traj:
    trials_to_analyse = ['go1_succ','go1_rtex', ]#'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
    sessions_to_analyse = ['bmi','sal']  #not looking at nan %
  
    all_traj_by_trial, _,= dlc_func.collect_traj_by_trialtype(trials_to_analyse, sessions_to_analyse,
                                                              mpfc_all_data, all_mismatched_files, scaled = False)
                 
    scaled_all_traj_by_trial, _ = dlc_func.collect_traj_by_trialtype(trials_to_analyse, sessions_to_analyse,
                                                                      mpfc_all_data, all_mismatched_files,scaled = True)    
    avg_all_norm_medians = dlc_func.get_avg_norm_medians(mpfc_all_data)

    
#%% plot all trials
if plot_traj:
    trials_to_plot = ['go1_succ','go1_rtex']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
    sessions_to_plot =  ['sal', 'bmi']
    traj_part = 'head'
    by_subject = True
    #subj_to_plot ='all'
    # subj_to_plot = [x[:-4] for x in list(mpfc_all_data.keys()) if 'rat01' in x]

    # dlc_func.plot_trajectories(trials_to_plot, sessions_to_plot,
    #                            traj_part, scaled_all_traj_by_trial, 
    #                            avg_all_norm_medians, subj_to_plot,by_subject)
    subj_to_plot = [x[:-4] for x in list(mpfc_all_data.keys()) if 'rat02' in x]

    dlc_func.plot_trajectories(trials_to_plot, sessions_to_plot,
                               traj_part, scaled_all_traj_by_trial, 
                               avg_all_norm_medians, subj_to_plot,by_subject)
    
#%%

num_traj = 50
plot_by_traj =True # single traj per fig
animals_to_plot = ['02_02']#,'02_16','02_17','02_11']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,mpfc_all_data,plot_by_traj)

#%%

f, ax = plt.subplots(1,1) 

a = flat_traj['go1_succ_sal_rat02_02_sal97'][0]

plt.plot(a['head'].x,a['head'].y)
dlc_func.plotbox(ax, mpfc_all_data['rat02_02_sal'].box_norm_medians)
#%%

check_frames  = 'yes'
start_frame = 0
end_frame = 200
frame_step = 5
frame_range = [start_frame,end_frame,frame_step]

manual = True
if check_frames == 'yes':
    tag_list = ['rat02_02_sal']
    vid_directory = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - MPFC/DLC_VIDEOS' 
    dlc_func.frame_checker(tag_list,vid_directory,[42860],frame_range,mpfc_all_data,'frame',manual)
     

#%% individual trial plotting

    # num_traj = 10
    # plot_by_traj =False # single traj per fig
    # animals_to_plot = ['06','13']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
    # dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,all_data,plot_by_traj)


#%% PDF - heat maaps


# NEED TO REALLY THINK ABOUT WHAT WE ARE PLOTTING HERE 
if plot_heat_maps:

    n_bins = 9
    
    all_pdfs = {}
    for tt in trials_to_plot:
        all_pdfs[tt] = {}
        for s in sessions_to_plot:
            session_data = all_traj_by_trial[tt][s]
            all_pdfs[tt][s] ={}
            for trial in session_data.keys():
                # all_pdfs[tt][s][trial],_,y_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                #                                    bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)))#,density =True) 
                density,_,y_ = np.histogram2d(session_data[trial][traj_part].x,session_data[trial][traj_part].y,
                                                    bins=(np.linspace(-150,150,n_bins+1),np.linspace(-0,200,n_bins+1)),density =True) 
                all_pdfs[tt][s][trial] = density/density.sum() # sum across all bins =  1
        #using a restricted map where area behaind the magazine is excluded
#%%      

def find_vmin_vmax(all_pdfs):
    vmin =1
    vmax = 0
    for tt in all_pdfs.keys():
        for s in all_pdfs[tt].keys(): 
            pdf_arr = np.array(list(all_pdfs[tt][s].values()))
            pdf_mean = pdf_arr.mean(axis=0)
            if np.max(pdf_mean)  > vmax:
                vmax = np.max(pdf_mean)
            else:
                vmax = vmax
            if np.min(pdf_mean) < vmin:
                vmin = np.min(pdf_mean)
            else:
                vmin = vmin
    vmax = np.ceil(vmax / 0.05) * 0.05
    vmin = np.floor(vmin / 0.05) * 0.05
    return vmax, vmin


#%%

#maybe get some measures of the number of trials in such tt x rew combination 


vmax, vmin = find_vmin_vmax(all_pdfs)

colors  = {'veh': 'Greens_r',
           '1mg': 'Oranges_r',
           '10mg': 'Reds_r'}
for trial_type in ['succ','rtex','wronglp']:
    print(trial_type)
    f, axs = plt.subplots(2,3, figsize = (20,12))
    for r in [1, 2]:
        i = 0 
        trial_rew_key = [x for x in all_pdfs.keys()
                         if trial_type in x and str(r) in x][0]
        print(trial_rew_key)
        for s in all_pdfs[trial_rew_key].keys(): 
            pdf_arr = np.array(list(all_pdfs[trial_rew_key][s].values()))
            pdf_mean = pdf_arr.mean(axis=0)
            if r == 2:
                pdf_mean = np.flip(pdf_mean, axis=0)
            masked = np.ma.masked_where(pdf_mean == 0 ,pdf_mean)
            cmap = colors[s]
            x = axs[r-1][i].imshow(masked, cmap = cmap,
                              vmax = vmax, vmin= vmin)                                               
            axs[r-1][i].set_title(s)
            axs[r-1][i].set_facecolor('k')
            axs[r-1][i].set_yticklabels('')
            axs[r-1][i].set_xticklabels('')
        
            plt.colorbar(x,ax = axs[r-1][i], fraction = 0.05)
            i += 1
           
        f.suptitle(trial_type)
        sns.set_style('white')
        sns.despine()

#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#   OCCUPANCY

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
Effectively draws this grid as adds out of bounds bins
points will only end in the 6-18 range


  0  |  1  |  2  |  3  |  4  
-----+-----+-----+-----+------
  5  |  6  |  7  |  8  |  9  
-----+-----+-----+-----+-----
  10  | 11  | 12  | 13  | 14  
-----+-----+-----+-----+-----
  15  | 16  | 17  | 18  | 19  
-----+-----+-----+-----+-----
  20  | 21  | 22  | 23  | 24  

prob want some kind of restriction- so that have to be in segment for more than x100ms to count 


11 = nosepoke
6 = small lever
16 = large lever
14 = food mag 
'''


plot_occ_checks = False
all_occ_ords, all_occ_traj = dlc_func.get_occupancy(all_traj_by_trial,'head')
all_occ_ords_scal = dlc_func.get_occupancy(scaled_all_traj_by_trial,'head')


if plot_occ_checks:
    # plot occupancy order over pdfs to check 
    tt = 'go1_rtex'
    s = '10mg'
    for trial in all_pdfs[tt][s].keys():                               
            f, ax = plt.subplots(1,1)
            ax.imshow(all_pdfs[tt][s][trial], cmap='hot')#,ax=ax)
            ax.set_title(all_occ_ords[tt][s][trial])
         
            
    # plotting occupancy order on top of trajectory plots 
    trials_to_plot = ['go1_rtex']# ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
    sessions_to_plot = ['veh']#,'1mg','10mg']
    num_traj = 10
    plot_by_traj =True
    animals_to_plot = ['06','13']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
    dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,
                                    animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,
                                    all_data,plot_by_traj,all_occ_ords, print_occ_ord = True)









#%%
flat_occ_ords = pd.json_normalize(all_occ_ords, sep='_').to_dict()
flat_occ_traj =  pd.json_normalize(all_occ_traj, sep='_').to_dict()
ecb_occ_analysis = dlc_func.occupancy_ord_struc(flat_occ_ords,flat_occ_traj,['veh','1mg','10mg'], all_mismatched_files)


#%%

corr_lever, corr_lever_m = ecb_occ_analysis.plot_all_proportions('corr_lever')
wrong_lever, _ = ecb_occ_analysis.plot_all_proportions('wrong_lever')
no_lever, _ = ecb_occ_analysis.plot_all_proportions('no_lever')
#%%
  

#%%

a = ecb_occ_analysis.test_proportions('corr_lever','rtex', plot= True)

a = ecb_occ_analysis.test_proportions('no_lever','rtex', plot= True)
a = ecb_occ_analysis.test_proportions('wrong_lever','rtex', plot= True)


visit_df = ecb_occ_analysis.find_1st_areas(plot = True)
#%%
def get_num_np_frames(k):
    t = flat_occ_traj[k][0]
    
    count = 0 
    for i in range(0,len(t)):
        if t[i] == 11:
            count += 1
        else:
            break
    return count



#%%

#   Need to check regularity of frame times!! 
#   are they all 0.04???

dlc_file_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/dlc analysis test dlc files'
all_frame_times = pickle.load( open(dlc_file_path + '/all_frametimes_dlc.p', 'rb'))
frame_rate_test = pd.DataFrame()
for key in all_frame_times.keys():
    frames = [float(x) for x in all_frame_times[key]]
    frame_rate_test.loc[key,'avg'] = np.mean(np.diff(frames))
    frame_rate_test.loc[key,'max'] = np.max(np.diff(frames))
    frame_rate_test.loc[key,'min'] = np.min(np.diff(frames))
print(frame_rate_test.loc[frame_rate_test['avg'] > 0.04])
    
#%%
df = ecb_occ_analysis.get_nosepoke_times()
plot = True
if plot == True:
    for trial_type in ['succ', 'rtex','wronglp']:
        plot_df = df.query('trial_type == @trial_type')
        plot_dfm = plot_df.melt(id_vars = ['trial_type','cond',
                                           'rew_size','animal'])
        x = sns.catplot(kind = 'bar', data = plot_dfm, 
                        x = 'trial_type',y = 'value', 
                        hue = 'cond', row = 'rew_size',
                        col = 'variable', palette = ecb_occ_analysis.palette, 
                        alpha = 0.7)
        x.fig.suptitle(trial_type)
        
#%%
fig, ax = plt.subplots(1,1)
dfm=  df.melt(id_vars = ['trial_type','cond',
                                   'rew_size','animal'],
              value_name = 'time_in_nosepoke_zone (s)')
x = sns.catplot(kind = 'bar', data = dfm, 
                 x = 'rew_size',y = 'time_in_nosepoke_zone (s)', 
                 hue = 'cond', 
                 row = 'trial_type', palette = ecb_occ_analysis.palette, 
                 alpha = 0.7)

# x = sns.catplot(kind = 'bar', data = dfm, 
#                  x = 'trial_type',y = 'time_in_nosepoke_zone (s)', 
#                  hue = 'cond', row = 'rew_size',
#                  col = 'variable', palette = ecb_occ_analysis.palette, 
#                  alpha = 0.7)

#%% calculating trajectory length - normalise by distance between levers

# CHECK need to crop traj so they end at the food magazine? maybe for correct trials - after succ trigger = go to food mag
# for incorrect trials  - after 5 sec timeout? 
traj_distance, mean_traj_distance = dlc_func.calc_traj_length(trials_to_plot,sessions_to_plot,all_traj_by_trial,avg_all_norm_medians,traj_part)

