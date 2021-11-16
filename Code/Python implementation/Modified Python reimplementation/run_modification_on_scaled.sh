#!/bin/bash

####-------------------------------------------------------------------------####

### Simulates the three algorithms on all the task variants using the modified Python reimplementation

### This script requires the json descriptors for the task and simulation parameteres to be provided in the Descriptions folder.

### To run: bash run_modification.sh

####-------------------------------------------------------------------------####

# Multiple simulations for each variant
for i in {1..10}
do

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg3_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg4_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg5_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg6_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg7_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg8_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg9_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg10_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg15_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg20_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg30_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg40_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg50_Var.json"

    python3 run.py --parameters="Descriptions_scaled/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions_scaled/task_parameter_file_Task2_ST_Seg100_Var.json"

done
