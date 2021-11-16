#!/bin/bash

####-------------------------------------------------------------------------####

### Simulates the three algorithms on all the task variants using the modified Python reimplementation

### This script requires the json descriptors for the task and simulation parameteres to be provided in the Descriptions folder.

### To run: bash run_modification.sh

####-------------------------------------------------------------------------####


# Multiple simulations for each variant
for i in {1..10}
do

    # Simulations of each algorithm on Task 1
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task1_FORCE.json" --experiment="Descriptions/task_parameter_file_Task1_FORCE.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task1_RMHL.json" --experiment="Descriptions/task_parameter_file_Task1_RMHL.json"
        python3 run.py --parameters="Descriptions/simulation_parameter_file_Task1_ST.json" --experiment="Descriptions/task_parameter_file_Task1_ST.json"

    # Simulations of each algorithm on Task 2
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task2_RMHL.json" --experiment="Descriptions/task_parameter_file_Task2_RMHL_Seg2.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task2_RMHL.json" --experiment="Descriptions/task_parameter_file_Task2_RMHL_Seg3_Var.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions/task_parameter_file_Task2_ST_Seg2.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task2_ST.json" --experiment="Descriptions/task_parameter_file_Task2_ST_Seg3_Var.json"

    # Simulations of each algorithm on Task 3
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task3_RMHL.json" --experiment="Descriptions/task_parameter_file_Task3_RMHL_05.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task3_RMHL.json" --experiment="Descriptions/task_parameter_file_Task3_RMHL.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task3_ST.json" --experiment="Descriptions/task_parameter_file_Task3_ST_05.json"
    python3 run.py --parameters="Descriptions/simulation_parameter_file_Task3_ST.json" --experiment="Descriptions/task_parameter_file_Task3_ST.json"

done

