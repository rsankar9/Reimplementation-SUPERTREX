#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Mar 24 19:12:15 2020
    @author: rsankar
    
    This script belongs to a modified reimplementation of the models described in -
    Pyle, R. and Rosenbaum, R., 2019.
    A reservoir computing model of reward-modulated motor learning and automaticity.
    Neural computation, 31(7), pp.1430-1461.
    
    This script builds the model object for the FORCE algorithm, trains and tests it on the described tasks and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os
import pandas as pd

class ModelFORCE():

    def __init__(s, parameters, task, exp):                                                 # self -> s
        """
            Initialise the model object.
            
            exp: dict
                Task description where:
                rseed           : seed for random generator; if rseed=0, a random seed is used
                dataset_file    : file to store task datapoints
                algorithm       : learning algorithm to simulate
                results_folder  : path to store results
                git-hash        : version of model being simulated
                timespan        : duration of 1 experiment trial
                task_type       : type of task, the model is to be run on
                n_segs          : no. of arm segments (irrelevant for task #1)
                arm_len         : length of each arm segment (irrelevant for task #1)
                arm_cost        : cost of moving each arm segment (irrelevant for task #1 and #2)
                display_plot    : show the plot, too, or just save it
                plot_format     : file format for saving plot (ps, eps, pdf, pgf, png, raw, rgba, svg, svgz, jpg, jpeg, tif, tiff)
            
            parameters: dict
                Parameter values where:
                    N               : no. of neurons in reservoir
                    lmbda           : controls spectral radius
                    sparsity        : connectivity sparsity in reservoir
                    dT              : time gradient in ms
                    n_train_trials  : no. of training trials
                    n_test_trials   : no. of testing trials
                    alpha           : attenuate noise
                    gamma           : Initialising factor for P matrix
                    k               : SUPERTREX learning rate
                    tau             : time constant of reservoir
                    tau_e           : low pass filter for MSE
                    tau_z           : low pass filter for z
            
            task: Task
                Task object created for this experiment
        """


        # Model parameters
        s.task_type         = exp['task_type']                                              # Type of task
        s.T                 = exp['timespan']                                               # Timescale of experiment in ms
        s.n_out             = exp['n_segs']                                                 # No. of arm segments
        if s.task_type == 1:    s.n_out = 2                                                 # or coordinates

        s.N                 = parameters['N']                                               # No. of neurons in reservoir
        s.lmbda             = parameters['lmbda']                                           # Related to spectral radius
        s.sparsity          = parameters['sparsity']                                        # Connectivity sparsity in reservoir
        s.dT                = parameters['dT']                                              # Time gradient
        s.n_train_trials    = parameters['n_train_trials']                                  # No. of trials for training > 5
        s.n_test_trials     = parameters['n_test_trials']                                   # No. of trials for testing
        s.n_total_trials     = parameters['n_train_trials'] + parameters['n_test_trials']   # No. of total trials
        s.alpha             = parameters['alpha']                                           # Noise attenuating factor
        s.gamma             = parameters['gamma']                                           # Initialising factor for P matrix
        s.tau               = parameters['tau']                                             # Time constant of reservoir leak
        s.tau_e             = parameters['tau_e']                                           # Low pass filter for displaying MSE
        s.tau_z             = parameters['tau_z']                                           # Low pass filter for displaying z

        s.leak              = s.dT/s.tau                                                    # Reservoir leak
        s.n_timesteps       = int(s.T/s.dT)                                                 # No. of timesteps in a trial
        s.sigma             = s.lmbda / np.sqrt((s.sparsity*s.N))                           # Standard deviation of initial reservoir connectivity

        if exp['rseed'] == 0:   s.rseed = np.random.randint(0,1e7)
        else:                   s.rseed = exp['rseed']                                      # Seed for randomisation
        print('Seed:', s.rseed)


        s.results_path      = exp['results_folder'] + '/' + str(s.rseed)  + '_nsegs'  + str(exp['n_segs']) + '/'

        if not os.path.exists(s.results_path):   os.makedirs(s.results_path)

        # Build reservoir architecture
        s.build(task)


    def build(s, task):
        """ Building the model architecture. """

        _data = task.data

        # Network initialisations
        np.random.seed(s.rseed)

        # Build reservoir
        s.J = np.zeros((s.N, s.N))
        Jne = task.round_up(s.N * s.N * s.sparsity)
        idx_x, idx_y = task.rand_int(s.N, Jne), task.rand_int(s.N, Jne)
        s.J[idx_x, idx_y] = 1
        Jnz, Jcnz = np.nonzero(s.J), np.count_nonzero(s.J)
        Jr = stats.norm.ppf(np.random.uniform(size=Jcnz))
        s.J[Jnz] = Jr[:Jcnz]
        s.J = s.J * s.sigma                                                                 # Reservoir connectivity strengths
        unnecessary = np.random.uniform(size=2090)                                          # Hard-coded for N=1000, to make it equivalent to matlab

        # Build network
        s.Q = (np.random.rand(s.n_out, s.N) * 2 - 1).T                                      # Reservoir feedback connectivity
        s.x = np.random.rand(s.N, 1) - .5 * np.ones((s.N, 1))                               # Reservoir voltages
        s.r = np.tanh(s.x)                                                                  # Reservoir activity
        s.z = np.zeros((s.n_out, 1))                                                        # Network output
        s.W_FORCE = np.zeros((s.n_out, s.N))                                                # FORCE readout weights


        # Training initialisations
        s.outputs = np.column_stack((_data['x'], _data['y']))
        s.outputs = s.outputs.reshape((s.outputs.shape[0], 2, 1))
        s.P = np.identity(s.N) / s.gamma                                                    # FORCE inverse correlation estimate initialization
        s.e = 0


        # Plotting purposes
        s.cost_rec    = np.zeros((s.n_total_trials, s.n_timesteps))
        s.error       = np.zeros((s.n_total_trials, s.n_timesteps))
        s.z_rec       = np.zeros((s.n_out, s.n_total_trials, s.n_timesteps, 1))
        s.z_FORCE_rec = np.zeros((s.n_out, s.n_total_trials, s.n_timesteps, 1))
        s.hz_rec      = np.zeros((2, s.n_total_trials, s.n_timesteps, 1))
        s.W_FORCE_rec = np.zeros((s.n_total_trials, s.n_timesteps))


    def train(s, task):
        """ Training the model using the FORCE algorithm. """

        # Online training
        print('Training')
        for trial_num in tqdm(range(s.n_train_trials)):
            for time_step in range(s.n_timesteps):

                # Update reservoir state
                s.x     = s.x + (s.leak)*(-s.x +np.dot(s.J,s.r) + np.dot(s.Q,s.z))
                xi_r    = np.random.uniform(0, 1, (s.N, 1)) * s.alpha * 2 - s.alpha
                s.r     = np.tanh(s.x) + xi_r

                # Compute output at current timestep
                z_FORCE = np.dot(s.W_FORCE,s.r)
                s.z     = z_FORCE
                hz      = task.h(s.z)

                # Computing error (In author's code, it's only calculated once every 10 timesteps)
                cost    = task.cost(s.z)
                ze      = hz-s.outputs[time_step]
                s.e     = np.sum(ze ** 2) + cost

                # Compute running estimate (every 10 timesteps to reduce computation time) (+1 because matlab is 1-indexed)
                if (time_step+1)%10 == 0:

                    Pr = np.dot(s.P,s.r)
                    rPr = np.dot(s.r.T,Pr)[0,0]
                    c = 1.0/(1.0 + rPr)
                    s.P -= np.dot(Pr, Pr.T * c)

                    s.W_FORCE += np.dot(c * -ze, Pr.T)

                # Recording purposes
                s.error[trial_num, time_step]           = s.e
                s.cost_rec[trial_num, time_step]        = cost
                s.hz_rec[:, trial_num, time_step]       = hz[:]
                s.z_rec[:, trial_num, time_step]        = s.z[:]
                s.z_FORCE_rec[:, trial_num, time_step]  = z_FORCE[:]
                s.W_FORCE_rec[trial_num, time_step]     = task.norm(s.W_FORCE)

        print('Training done')


    def test(s, task):
        """ Testing the stability of the FORCE algorithm. """

        # Testing
        print('Testing')
        for trial_num in tqdm(range(s.n_train_trials, s.n_total_trials)):
            for time_step in range(s.n_timesteps):

                # Update reservoir state
                zt  = s.outputs[time_step] #s.z_rec[:,trial_num-5,time_step]
                s.x = s.x + (s.leak)*(-s.x +np.dot(s.J,s.r) + np.dot(s.Q,zt))
                s.r = np.tanh(s.x)

                # Compute output at current timestep
                z_FORCE = np.dot(s.W_FORCE,s.r)
                s.z     = z_FORCE
                hz      = task.h(s.z)

                # Computing error (in author's code?)
                cost    = task.cost(s.z)
                ze      = hz-s.outputs[time_step]
                s.e     = np.sum(ze ** 2) + cost

                # Recording purposes
                s.error[trial_num, time_step]           = s.e
                s.cost_rec[trial_num, time_step]        = cost
                s.hz_rec[:, trial_num, time_step]       = hz[:]
                s.z_rec[:, trial_num, time_step]        = s.z[:]
                s.z_FORCE_rec[:, trial_num, time_step]  = z_FORCE[:]


    def save_results(s, exp):
        """ Saves the results of the simulation. """

        print('Saving results')
        np.savez(s.results_path + 'Data',
                    error               = s.error,
                    cost                = s.cost_rec,
                    z                   = s.z_rec,
                    z_FORCE             = s.z_FORCE_rec,
                    hz                  = s.hz_rec,
                    W_FORCE             = s.W_FORCE_rec
                    )
    
    
    def plot(s, exp, task):
        """
            Loads the saved results and plots an overall figure.
        """
        
        # Load dataset
        data = np.load(exp['dataset_file'])
        target_coord = np.array((np.tile(data['x'], s.n_train_trials+s.n_test_trials),
        np.tile(data['y'], s.n_train_trials+s.n_test_trials)))

        # Load result arrays
        _ = np.load(s.results_path + 'Data.npz')
        _hz = _['hz']
        _z = _['z']
        _z_FORCE = _['z_FORCE']
        _W_FORCE = _['W_FORCE']
        _error = _['error']
        _cost = _['cost']
         
        # Low pass filter results
        _cost = _cost.flatten()
        _error = _error.flatten()
        _z = np.reshape(_z, (s.n_out, _z[0].size))
        _W_FORCE = _W_FORCE.flatten()
        _z_FORCE = np.reshape(_z_FORCE, (s.n_out, _z_FORCE[0].size))
        _hz = np.reshape(_hz, (2, _hz[0].size))
        
        cost_bar = np.copy(_cost)
        mse_bar = np.copy(_error)
        z_bar = np.copy(_z)
        z_FORCE_bar = np.copy(_z_FORCE)
        hz_bar = np.copy(_hz)
        ce = s.dT / s.tau_e
        cz = s.dT / s.tau_z
        
        for i in np.arange(1, s.n_timesteps * s.n_total_trials):
            z_bar[:, i]         = z_bar[:, i - 1]       + cz * (-z_bar[:, i - 1]        + _z[:, i])
            z_FORCE_bar[:, i]   = z_FORCE_bar[:, i - 1] + cz * (-z_FORCE_bar[:, i - 1]  + _z_FORCE[:, i])
            hz_bar[:, i]        = hz_bar[:, i - 1]      + cz * (-hz_bar[:, i - 1]       + _hz[:, i])
            mse_bar[i]          = mse_bar[i - 1]        + ce * (-mse_bar[i - 1]         + _error[i])
            cost_bar[i]         = cost_bar[i - 1]       + ce * (-cost_bar[i - 1]        + _cost[i])
        mse = np.sqrt(mse_bar)

        mean_sqrtmsebar = np.mean(mse[s.n_timesteps * s.n_train_trials:])
        std_sqrtmsebar = np.std(mse[s.n_timesteps * s.n_train_trials:])
        median_sqrtmsebar = np.median(mse[s.n_timesteps * s.n_train_trials:])
        test_stats = {
            'mean':     [mean_sqrtmsebar],
            'median':   [median_sqrtmsebar],
            'std':      [std_sqrtmsebar]
        }
        df = pd.DataFrame(test_stats)
        df.to_csv(s.results_path + 'test_stats.csv', index=False)
    
        # Adjusting for uncalculated W_FORCE norms and NANs/INF while calculating norm
        for i in np.arange(1,s.n_timesteps * s.n_total_trials):
            if _W_FORCE[i] == 0:    _W_FORCE[i] = _W_FORCE[i-1]
            
        # ------------------------------------------------------------------- #
        # Plot
        print('Plotting')
        
        n_subplots = 5                                                   # For timeseries output, norm and error, x and y coordinates

        fig, ax = plt.subplots(n_subplots)
        fig.suptitle('Results of FORCE simulation on Task #' + str(task.type) + ' with ' + str(exp['n_segs']) + ' segments at seed: ' + str(s.rseed))
        
        ax[0].set_title('Output during testing phase')
        ax[0].set(aspect='equal')
        l1 = ax[0].plot(data['x'], data['y'], marker=',', color='red', markersize=1)
        sp, ep = int(s.n_timesteps * s.n_train_trials), int(s.n_timesteps * s.n_total_trials)
        ax[0].plot(hz_bar[0, sp:ep], hz_bar[1, sp:ep], marker=',', markersize=2, color='orange')
        
        ax[1].set_title('Norm of weight matrix')
        ax[1].plot(_W_FORCE.flatten(), color='orange')
        ax[1].set_ylabel('||W||')
        ax[1].axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=4, alpha=0.5)
        
        ax[2].set_title('Distance from target')
        ax[2].plot(mse[::10], color='orange')
        ax[2].set_ylabel('E')
        l2 = ax[2].axvline(x=(s.n_train_trials * s.n_timesteps)/10, color='grey', linewidth=4, alpha=0.5)
        ax[2].axhline(y=mean_sqrtmsebar, xmin=2/3, xmax=0.95, color='grey', linewidth=1, alpha=0.5)
        ax[2].text(s.n_total_trials*s.n_timesteps/10, mean_sqrtmsebar, '{:.2e}'.format(mean_sqrtmsebar), fontsize=12, va='center_baseline', ha='left', alpha=0.7, color='grey')
        ax[2].set_yscale('log')
        ax[2].set_yticks([1e-6, 1e-4, 1e-2, 1])

        ax[3].set_title('Coordinates')
        ax[3].set_ylabel('x')
        ax[3].plot(hz_bar[0], color='purple')
        ax[3].plot(target_coord[0], color='red')
        ax[3].axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=4, alpha=0.5)
        
        ax[4].set_ylabel('y')
        ax[4].plot(hz_bar[1], color='purple')
        ax[4].plot(target_coord[1], color='red')
        ax[4].axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=4, alpha=0.5)
        
        lines = [ax[0].plot(1, 1, color='purple')[0], ax[0].plot(1, 1, color='green')[0],
        ax[0].plot(1, 1, color='orange')[0], l1, l2]
        labels = ['SUPERTREX', 'Exploratory', 'Mastery', 'Target', 'Test phase']
        fig.legend(lines, labels)
        
        for k in range(n_subplots):
            ax[k].spines['top'].set_visible(False)
            ax[k].spines['right'].set_visible(False)
            ax[k].spines['bottom'].set_visible(False)
            ax[k].get_xaxis().set_ticks([])
        
        plt.savefig(s.results_path + 'Overall.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))
        
        if exp['display_plot'] == 'Yes':
            fig.canvas.manager.window.showMaximized()
            plt.show()

        print('Done.')
             # ------------------------------------------------------------------- #



    def plot_distinct(s, exp, task):
        """
            Loads the saved results and plots individual figures.
        """

        # Load dataset
        data = np.load(exp['dataset_file'])
        target_coord = np.array((np.tile(data['x'], s.n_train_trials+s.n_test_trials),
                                   np.tile(data['y'], s.n_train_trials+s.n_test_trials)))

        # Load result arrays
        _ = np.load(s.results_path + 'Data' + '.npz')
        _hz = _['hz']
        _z = _['z']
        _z_FORCE = _['z_FORCE']
        _W_FORCE = _['W_FORCE']
        _error = _['error']
        _cost = _['cost']

        # Low pass filter results
        _cost = _cost.flatten()
        _error = _error.flatten()
        _z = np.reshape(_z, (s.n_out, _z[0].size))
        _W_FORCE = _W_FORCE.flatten()
        _z_FORCE = np.reshape(_z_FORCE, (s.n_out, _z_FORCE[0].size))
        _hz = np.reshape(_hz, (2, _hz[0].size))

        cost_bar = np.copy(_cost)
        mse_bar = np.copy(_error)
        z_bar = np.copy(_z)
        z_FORCE_bar = np.copy(_z_FORCE)
        hz_bar = np.copy(_hz)
        ce = s.dT / s.tau_e
        cz = s.dT / s.tau_z

        for i in np.arange(1, s.n_timesteps * s.n_total_trials):
            z_bar[:, i]         = z_bar[:, i - 1]       + cz * (-z_bar[:, i - 1]        + _z[:, i])
            z_FORCE_bar[:, i]   = z_FORCE_bar[:, i - 1] + cz * (-z_FORCE_bar[:, i - 1]  + _z_FORCE[:, i])
            hz_bar[:, i]        = hz_bar[:, i - 1]      + cz * (-hz_bar[:, i - 1]       + _hz[:, i])
            mse_bar[i]          = mse_bar[i - 1]        + ce * (-mse_bar[i - 1]         + _error[i])
            cost_bar[i]         = cost_bar[i - 1]       + ce * (-cost_bar[i - 1]        + _cost[i])
        mse = np.sqrt(mse_bar)

        mean_sqrtmsebar = np.mean(mse[s.n_timesteps * s.n_train_trials:])
        std_sqrtmsebar = np.std(mse[s.n_timesteps * s.n_train_trials:])
        median_sqrtmsebar = np.median(mse[s.n_timesteps * s.n_train_trials:])
        test_stats = {
            'mean':     [mean_sqrtmsebar],
            'median':   [median_sqrtmsebar],
            'std':      [std_sqrtmsebar]
        }
        df = pd.DataFrame(test_stats)
        df.to_csv(s.results_path + 'test_stats.csv', index=False)


        # Adjusting for uncalculated W_FORCE norms and NANs/INF while calculating norm
        for i in np.arange(1,s.n_timesteps * s.n_total_trials):
            if _W_FORCE[i] == 0:    _W_FORCE[i] = _W_FORCE[i-1]

        # ------------------------------------------------------------------- #
        # Plot
        print('Plotting')

        n_plots = 5                                                   # For timeseries output, norm and error, x and y coordinates

        fig, ax = plt.subplots(1)
        
#        ax[0].set_title('Output during testing phase')
        ax.set(aspect='equal')
        sp, ep = int(s.n_timesteps * s.n_train_trials), int(s.n_timesteps * s.n_total_trials)
        ax.plot(hz_bar[0, sp:ep:10], hz_bar[1, sp:ep:10], marker=',', markersize=0.5, color='blue')
        l1 = ax.plot(data['x'], data['y'], marker=',', color='red', markersize=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        plt.savefig(s.results_path + 'TimeSeries.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))


        fig, ax = plt.subplots(1)
        
#        ax.set_title('Norm of weight matrix')
        ax.plot(_W_FORCE.flatten(), color='grey', linewidth=0.5)
        ax.set_ylabel('||W||')
        ax.axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=2, alpha=0.5)
        ax.set_ylim(0, 0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])

        plt.savefig(s.results_path + 'W_norm.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))

        
        fig, ax = plt.subplots(1)

#        ax.set_title('Distance from target')
        ax.plot(mse[::10], color='blue', linewidth=0.5)
        ax.set_ylabel('Distance from Target')
        l2 = ax.axvline(x=(s.n_train_trials * s.n_timesteps)/10, color='grey', linewidth=2, alpha=0.5)
        ax.axhline(y=mean_sqrtmsebar, xmin=2/3, xmax=0.95, color='grey', linewidth=1, alpha=0.5)
        ax.text(s.n_total_trials*s.n_timesteps/10, mean_sqrtmsebar, '{:.2e}'.format(mean_sqrtmsebar), fontsize=12, va='center_baseline', ha='left', alpha=0.7, color='grey')
        ax.set_yscale('log')
        ax.get_yaxis().set_ticks([1e-6, 1e-4, 1e-2, 1])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        
        plt.savefig(s.results_path + 'MSE.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))


        fig, ax = plt.subplots(1)

#        ax.set_title('Coordinates')
        ax.set_ylabel('x(t)')
        ax.plot(hz_bar[0], color='blue', linewidth=0.5)
        ax.plot(target_coord[0], color='red', linewidth=0.5)
        ax.axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=2, alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        plt.savefig(s.results_path + 'CoordinateX.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))

        
        fig, ax = plt.subplots(1)

#        ax.set_title('Coordinates')
        ax.set_ylabel('y(t)')
        ax.plot(hz_bar[1], color='blue', linewidth=0.5)
        ax.plot(target_coord[1], color='red', linewidth=0.5)
        ax.axvline(x=s.n_train_trials * s.n_timesteps, color='grey', linewidth=2, alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        plt.savefig(s.results_path + 'CoordinateY.' + exp['plot_format'], rasterized=(exp['plot_format']=='pdf'))

        print('Done.')
        # ------------------------------------------------------------------- #
