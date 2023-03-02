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

#%% functions

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def get_occupancy(all_traj_by_trial,traj_part):  
    all_occ_ords = {}
    all_occ_traj = {}
    bin_2d_dict = {}
    bin_edge_x = [-150.,  -50.,   50.,  150.]
    bin_edge_y = [  0.        ,  66.66666667, 133.33333333, 200.        ]
    for tt in all_traj_by_trial.keys():
        bin_2d_dict[tt] = {}
        all_occ_ords[tt] = {}
        all_occ_traj[tt] = {}
        for s in all_traj_by_trial[tt].keys():
            session_data = all_traj_by_trial[tt][s]
            bin_2d_dict[tt][s] ={}
            all_occ_ords[tt][s] = {}
            all_occ_traj[tt][s] = {}
            for trial in session_data.keys():
                b = st.binned_statistic_2d(session_data[trial][traj_part].x,
                                                           session_data[trial][traj_part].y,
                                                          None,'count',
                                                          bins = [bin_edge_x, bin_edge_y])
                if all(val in [6,7,8,11,12,13,16,17,18] for val in b.binnumber) == False:
                    print(trial)  
                else:
                    print('*')
                bin_2d_dict[tt][s][trial] = b
                all_occ_ords[tt][s][trial] =  np.hstack([b.binnumber[0] , [b.binnumber[val] 
                                                       for val in range(1,len(b.binnumber)) 
                                                       if b.binnumber[val] != b.binnumber[val-1]]])
                all_occ_traj[tt][s][trial] = b.binnumber
    return all_occ_ords, all_occ_traj



    
            
class occupancy_ord_struc:
    def __init__(self, occ_ord_dict, occ_traj_dict, conditions, mismatched_files):
        self.occ_dict = occ_ord_dict #dict with all square occupany orders for all trials (flattened)
        self.traj_dict = occ_traj_dict
        self.conditions = conditions #eg veh dose1 dose2
        self.all_keys = list(self.occ_dict.keys())
        self.animals = list(set([k[k.find('rat') :k.rfind('_')]
                for k in self.all_keys]))
        self.mismatched_files = mismatched_files
        self.palette = ['darkgray','yellowgreen', 'forestgreen'] # palette for plotting
        
    def no_lever_occ(self,k):
        # checks if neither lever was entered on that trial
        if 6 not in self.occ_dict[k][0] and 16 not in self.occ_dict[k][0]:
            return k
        
    def small_lever_occ(self,k):
        # checks if small lever was entered on that trials
          if 6 in self.occ_dict[k][0]:
              return k
          
    def large_lever_occ(self,k):
        #checks if large lever was entered on that trial 
        if 16 in self.occ_dict[k][0]:
               return k
      
    def find_proportions(self,trial_type,func):
        '''
        finds propotions of trials of trial type x (eg go small success)
        that have entry into square defined by func(eg no_lever_occ)
        for each animal
        
        if no trials for that type exists for that animal records a nan 
        '''
        f = getattr(self,func)
        df = pd.DataFrame(columns = self.conditions)
        for cond in self.conditions:
            for animal in self.animals:
                animal_cond_keys = [k for k in self.all_keys 
                                    if trial_type in k 
                                    and cond in k 
                                    and animal in k]
                filt_keys = [f(k) for k in animal_cond_keys
                             if f(k) is not None]
                print(animal_cond_keys)
                if len(filt_keys) > 0:
                    df.loc[animal,cond] = len(filt_keys)/len(animal_cond_keys)
                elif animal + '_' + cond in self.mismatched_files:
                    df.loc[animal,cond] = np.nan
                elif len(animal_cond_keys) == 0:
                    df.loc[animal,cond] = np.nan
                else:
                    df.loc[animal,cond] = 0
        self.prop_df = df
        self.prop_df_title = (trial_type + ' ' + func)
        return df
    
    # def plot_proportions(self):
    #     dfm = self.prop_df.melt()
    #     fig, ax = plt.subplots(1,1)
    #     sns.barplot(data = dfm,
    #          x ='variable',
    #          y = 'value',
    #          ax = ax)
    #     sns.stripplot(data = dfm,
    #          x ='variable',
    #          y = 'value',
    #          color = 'black',
    #          alpha = 0.7,
    #          ax = ax)
    #     ax.set_title(self.prop_df_title)
    
    def plot_all_proportions(self, filt):
        
        '''
        finds and plots the proportion of trials for all trial types that 
        have entry into the sqaure defined by filt (eg correct lever square)
        '''
        
        filt_dict = {'corr_lever': ['small_lever_occ','large_lever_occ'],
                     'wrong_lever': ['large_lever_occ','small_lever_occ'], 
                     'no_lever': ['no_lever_occ','no_lever_occ']}
        
        
        output_df = pd.DataFrame()
        for trial_type in  ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']:
            if 'go1' in trial_type:
                df = self.find_proportions(trial_type, filt_dict[filt][0])
                df.loc[:,'trial_type'] = trial_type
                output_df = pd.concat([output_df, df])
            else:
                df = self.find_proportions(trial_type, filt_dict[filt][1])
                df.loc[:,'trial_type'] = trial_type
                output_df = pd.concat([output_df, df])
          
        output_df.loc[:,'rew_size'] =  [x[2] for x in output_df.trial_type]
        output_df.loc[:,'trial_type'] =  [x[4:] for x in output_df.trial_type]
        scatter_palette = [adjust_lightness(x) for x in self.palette]
        output_dfm = output_df.melt(id_vars=['trial_type', 'rew_size'])
        output_dfm.loc[:,'value'] = output_dfm.value.astype('float')
        x = sns.FacetGrid(output_dfm,
                          row = 'rew_size',
                           margin_titles =True )
        x.map_dataframe(sns.barplot,x="trial_type",
                        y="value",hue='variable'
                       , palette =self.palette,
                       errorbar = None)
        x.map_dataframe(sns.stripplot,x="trial_type", 
                        y="value", hue='variable',
                          alpha= 0.6 ,palette = scatter_palette,
                          dodge=True, jitter=False)
        
   
        x.set_axis_labels("Dose", "Proportion of trials")
        x.fig.suptitle(filt+' sqaure entry', x = 0.75)
        x.fig.set_size_inches(8.5, 5.5)
        x.set(ylim=(0, 1))

        setattr(self, filt + '_df', output_df)
        return output_df, output_dfm
            
    
    # '''
    # finds and plots the proportion of trials for all trial types that 
    # have entry into the sqaure defined by filt (eg correct lever square)
    # '''
    
    # filt_dict = {'corr_lever': ['small_lever_occ','large_lever_occ'],
    #              'wrong_lever': ['large_lever_occ','small_lever_occ'], 
    #              'no_lever': ['no_lever_occ','no_lever_occ']}
    
    
    # output_df = pd.DataFrame()
    # for trial_type in  ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']:
    #     if 'go1' in trial_type:
    #         df = self.find_proportions(trial_type, filt_dict[filt][0])
    #         df.loc[:,'trial_type'] = trial_type
    #         output_df = pd.concat([output_df, df])
    #     else:
    #         df = self.find_proportions(trial_type, filt_dict[filt][1])
    #         df.loc[:,'trial_type'] = trial_type
    #         output_df = pd.concat([output_df, df])
      
    # output_df.loc[:,'rew_size'] =  [x[2] for x in output_df.trial_type]
    # output_df.loc[:,'trial_type'] =  [x[4:] for x in output_df.trial_type]
    
    # output_dfm = output_df.melt(id_vars=['trial_type', 'rew_size'])
    # output_dfm.loc[:,'value'] = output_dfm.value.astype('float')
    # x = sns.catplot(kind = 'bar', data = output_dfm, x = 'trial_type',
    #                 y = 'value', hue = 'variable', row = 'rew_size', 
    #                 palette = self.palette, alpha = 0.7)
    # x.set_axis_labels("Dose", "Proportion of trials")
    # x.fig.suptitle(filt+' sqaure entry', x = 0.75)
    # x.fig.set_size_inches(8.5, 5.5)
    # x.set(ylim=(0, 1))

    # setattr(self, filt + '_df', output_df)
    # return output_df, output_dfm
    def test_proportions(self, filt, trial_type, plot= False):      
        '''
        performs anova on results of "plot all proportios"
        Parameters
        ----------
        filt : str
            condition to be test eg corr_lever .
        trial_type : str
            .
        plot : TYPE, optional
            plots a bar chart
        Returns
        -------
        anova : TYPE
            DESCRIPTION.

        '''
        df = getattr(self, filt + '_df')
        df.loc[:,'subj'] = df.index
        anova_dfm = df.query('trial_type == @trial_type').melt(id_vars = ['trial_type',
                                                               'rew_size',
                                                               'subj'],
                                                    var_name = 'dose', 
                                                    value_name ='prop_of_trials')
        anova_dfm.loc[:,'prop_of_trials'] = anova_dfm.prop_of_trials.astype('float')
        anova_dfm.loc[anova_dfm['prop_of_trials'] == np.nan] = 0 # nans are no errors of that tyep - so prop stioll should be zero? 
        anova = pg.rm_anova(data = anova_dfm, dv = 'prop_of_trials', 
                            within = ['rew_size','dose'],
                            subject = 'subj', detailed=True, 
                            effsize="np2")
        print(anova)
        if plot == True:
            fig, axs = plt.subplots(1,2, sharey =True, figsize =(4,5))
            anova_dfm.rew_size = anova_dfm.rew_size.astype('int')
            for r in range(0,2):
                sns.barplot(data = anova_dfm.query('rew_size == @r+1'),
                            x = 'dose',
                            y = 'prop_of_trials', 
                            palette = self.palette,
                            ax = axs[r], errorbar = None)
                sns.stripplot(data = anova_dfm.query('rew_size == @r+1'),
                             x = 'dose',
                             y = 'prop_of_trials', 
                             ax = axs[r], 
                             color = 'k', alpha =0.6)
                axs[r].set_title('Reward = ' + str(r+1))
                sns.despine()
                fig.suptitle(f'Prop of {filt} on {trial_type}')
        return anova
    
    def get_1st_area(self, key): 
        
        # drop all apart from levers and foodmag 
        rel_areas = [x for x in self.occ_dict[key][0]
                     if x in [6, 16, 13]]
        if len(rel_areas) > 0:
            val = rel_areas[0]
        else: 
            val = 100
        return val 
    
    def find_1st_areas(self, plot = False):
        go1_map = {6: 'corr_lever',
                   16: 'wrong_lever',
                   13: 'food_mag',
                   100: 'no_entry'}

        go2_map = {6: 'wrong_lever',
                   16: 'corr_lever',
                   13: 'food_mag',
                   100: 'no_entry'}
        
        df = pd.DataFrame(columns = ['corr_lever','wrong_lever','food_mag',
                                     'no_entry','trial_type','cond'])
        obs = 0
        for trial in ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']:
            for cond in self.conditions:
                for animal in self.animals:
                    if animal + '_' + cond not in self.mismatched_files:
                        animal_cond_keys = [k for k in self.all_keys 
                                            if trial in k 
                                            and cond in k 
                                            and animal in k]
                        list_1st_area = pd.DataFrame([self.get_1st_area(x) for x in animal_cond_keys], columns =['area'])
                        if 'go1' in trial:
                            list_1st_area['area'] = list_1st_area['area'].map(go1_map)
                        else:
                            list_1st_area['area'] = list_1st_area['area'].map(go2_map)
                        list_1st_area['area'] = list_1st_area['area'].astype('category')
                        list_1st_area['area'] = list_1st_area['area'].cat.set_categories(['corr_lever','wrong_lever',
                                                                                          'food_mag','no_entry'])
                        df.loc[obs,['corr_lever','wrong_lever',
                                    'food_mag','no_entry']] = list_1st_area['area'].value_counts(normalize = True)
                        df.loc[obs,'trial_type'] = trial[4:]
                        df.loc[obs,'cond'] = cond
                        df.loc[obs,'animal'] = animal
                        df.loc[obs,'rew_size'] = trial[2]
                        obs += 1
        if plot == True:
            for trial_type in ['succ', 'rtex','wronglp']:
                plot_df = df.query('trial_type == @trial_type')
                plot_dfm = plot_df.melt(id_vars = ['trial_type','cond',
                                                   'rew_size','animal'])
                plot_dfm.loc[:,'value'] = plot_dfm.value.astype('float')
                x = sns.FacetGrid(plot_dfm, col = 'rew_size', row = 'variable') 
                
                x.map_dataframe(sns.barplot,x="trial_type", y="value",
                                hue='cond', errorbar = None,
                                palette = ecb_occ_analysis.palette)
                
                x.map_dataframe(sns.stripplot,x="trial_type", y="value",
                                hue='cond',alpha= 0.6 ,color = 'black',
                                dodge=True, jitter=False)
              
                                
            
                x.fig.suptitle(trial_type)
        return df
    
    def get_num_np_frames(self,k):
        t = self.occ_traj[k][0]
        count = 0 
        for i in range(0,len(t)):
            if t[i] == 11:
                count += 1
            else:
                break
        return count
                    
    def get_nosepoke_times(self):
        df = pd.DataFrame()
        obs = 0
        for trial in ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']:
            for cond in self.conditions:
                for animal in self.animals:
                    if animal + '_' + cond not in self.mismatched_files:
                        animal_cond_keys = [k for k in self.all_keys 
                                            if trial in k 
                                            and cond in k 
                                            and animal in k]
                        nosepoke_time_s =  np.mean([get_num_np_frames(k) for k in animal_cond_keys]) * 0.04
                        df.loc[obs,'avg_np_time'] = nosepoke_time_s
                        df.loc[obs,'trial_type'] = trial[4:]
                        df.loc[obs,'cond'] = cond
                        df.loc[obs,'animal'] = animal
                        df.loc[obs,'rew_size'] = trial[2]
                        obs += 1
        return df



#%%
experiment = 'ecb'


data_path = 'C:/Users/George/OneDrive - Nexus365/Documents/GNG - ABD/'


all_data =             pickle.load(open (data_path + 'all_data_' +  experiment + '_preproc.p', 'rb'))
all_mismatched_files = pickle.load(open (data_path + 'mismatched_files' + experiment + '.p', 'rb'))

all_mismatched_files = all_mismatched_files + ['rat20_1mg']

extract_traj = True
plot_traj = True
plot_heat_maps = True

#%%

if extract_traj:
    trials_to_analyse = ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
    sessions_to_analyse = ['veh','1mg','10mg']
    #not looking at nan %
    all_traj_by_trial, _,= dlc_func.collect_traj_by_trialtype(trials_to_analyse, sessions_to_analyse,
                                                              all_data, all_mismatched_files, scaled = False)
                 
    scaled_all_traj_by_trial, _ = dlc_func.collect_traj_by_trialtype(trials_to_analyse, sessions_to_analyse,
                                                                      all_data, all_mismatched_files,scaled = True)    
    avg_all_norm_medians = dlc_func.get_avg_norm_medians(all_data)

    
#%% plot all trials
if plot_traj:
    trials_to_plot = ['go1_succ','go1_rtex', 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']# 'go1_wronglp','go2_succ', 'go2_rtex', 'go2_wronglp']
    sessions_to_plot = ['veh','1mg','10mg']#['veh','1mg','10mg']
    traj_part = 'head'
    by_subject = True
    subj_to_plot ='all'
    dlc_func.plot_trajectories(trials_to_plot, sessions_to_plot,
                               traj_part, scaled_all_traj_by_trial, 
                               avg_all_norm_medians, subj_to_plot,by_subject)


#%% individual trial plotting

    num_traj = 1
    plot_by_traj =False # single traj per fig
    animals_to_plot = ['06','13']#,'06','07','08','09']#,'11','12','13','14','19','20','21','22','23','24']
    dlc_func.plot_indv_trajectories(trials_to_plot, sessions_to_plot,animals_to_plot,traj_part,all_traj_by_trial,avg_all_norm_medians,num_traj,all_data,plot_by_traj)


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
               # all_pdfs[tt][s][trial] = density/density.sum() # sum across all bins =  1
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
all_occ_ords, all_occ_traj = get_occupancy(all_traj_by_trial,'head')
all_occ_ords_scal = get_occupancy(scaled_all_traj_by_trial,'head')


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
ecb_occ_analysis = occupancy_ord_struc(flat_occ_ords,flat_occ_traj,['veh','1mg','10mg'], all_mismatched_files)


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
        x = sns.FacetGrid(plot_dfm, col="variable", row = 'rew_size',
                           margin_titles =False )
        x.map_dataframe(sns.barplot,x="trial_type", y="value",
                        hue='cond', errorbar = None,
                        palette = ecb_occ_analysis.palette)
        x.map_dataframe(sns.stripplot,x="trial_type", y="value",
                        hue='cond',
                       
                          alpha= 0.6 ,color = 'black',
                          dodge=True, jitter=False)
        
        x.fig.subplots_adjust(top=0.9) # adjust the Figure in rp

        x.fig.suptitle(trial_type)
#%%
rtex_np_zone = df.query('trial_type == "rtex"')
anova = pg.rm_anova(data = rtex_np_zone, dv = 'avg_np_time', 
                      within = ['rew_size','cond'],
                      subject = 'animal', detailed=True, 
                      effsize="np2")

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

