#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Mar 24 19:12:15 2020
    @author: rsankar
    
    This script belongs to a modified reimplementation of the models described in -
    Pyle, R. and Rosenbaum, R., 2019.
    A reservoir computing model of reward-modulated motor learning and automaticity.
    Neural computation, 31(7), pp.1430-1461.
    
    This script creates the task and model objects as per the json descriptors.
    
"""

from ModelFORCE import ModelFORCE
from ModelRMHL import ModelRMHL
from ModelSUPERTREX import ModelSUPERTREX
from Task import Task


class Experiment():
    """ This object holds the description about the current simulation."""
    
    def __init__(self, exp, parameters):
        """ Initialize the experiment object. """
        
        # Create Task and Model objects
        self.task  = Task(exp, parameters)
        self.model = self.Model(exp, parameters)

        
    def Model(self, exp, parameters):
        """ This function build a Model object depending on the learning algorithm. """
    
        _algo = exp['algorithm']
        
        if _algo == 'FORCE':        return ModelFORCE(parameters, self.task, exp)
        elif _algo == 'RMHL':       return ModelRMHL(parameters, self.task, exp)
        elif _algo == 'SUPERTREX':  return ModelSUPERTREX(parameters, self.task, exp)

        
    def run(self, exp):
        """
            This function runs the experiment,
            by training and testing the model on the task,
            and saving the results.
        """
        self.model.train(self.task)
        self.model.test(self.task)
        self.model.save_results(exp)
        
        
    def plot(self, exp):
        """ This function reroutes to the appropriate plot function, as per the task type."""
        
        self.model.plot(exp, self.task)                 # Plots 1 overall figure with all the information; Can be commented out
        self.model.plot_distinct(exp, self.task)        # Plots individual figures; Can be commented out

