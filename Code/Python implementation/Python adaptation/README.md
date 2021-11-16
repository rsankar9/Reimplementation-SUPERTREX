#### README.md

Ref: 
Pyle, R. and Rosenbaum, R., 2019.
A reservoir computing model of reward-modulated motor learning and automaticity.
Neural computation, 31(7), pp.1430-1461.

This folder contains a Python implementation of the models presented in the  abovementioned paper in Python.

The re-implementation has been submitted to [ReScience C](https://rescience-c.github.io/),  in collaboration with Nicolas Thou, Nicolas Rougier and Arthur Leblois.

Author: Remya Sankar

#### Contents

2 directories:-

- ```Descriptions```: Contains the json descriptor files with the task and simulation parameters for each task and algorithm. These can be modified to test variants of tasks and hyper-parameters without altering the scripts. Note: In the task descriptor file, using rseed "0" leads to a random seed being chosen.  Using any other number, leads to that particular number being used as the random seed.
- ```Results```: Contains the results for each task variant. For each task variant, the simulation results using the default seed 5489 and additional 10 arbitrary seeds for the random number generator have been provided.

1 bash script:-

-   ```run_adaptation.sh```: Simulates each algorithm on all variants using the task and simulation parameters from the folder 'Descriptions' and stores the results in the folder 'Results'.

6 python scripts:-

- ```run.py```: Loads the descriptor files
- ```Experiment.py```: Creates the task and model objects
- ```Task.py```: Contains the task-specific functions used commonly across the models
- ```ModelFORCE.py```: Simulates and plots the FORCE algorithm on any task.
- ```ModelRMHL.py```: Simulates and plots the RMHL algorithm on any task.
- ```ModelSUPERTREX.py```: Simulates and plots the SUPERTREX algorithm on any task.

1 dataset file:-

- ```butterfly_coords.npz```: Contains the target timeseries. This is generated automatically while simulating any task.

1 dependencies file:-

- ```requirements.txt```: Contains information about the required dependencies.

##### Usage

-  To run as a batch, use the script ```run_adaptation.sh``` to generate results of the three algorithms on all task variants. This corresponds to Figure 1, 2, 3 and 4 of the paper about the re-implementation.
- As a default, to test the FORCE algorithm on Task 1:  ```python3 run.py```
-  To test individually or to run your own variant: ```python3 run.py
--parameters="<Path_to_simulation_parameter_file.json>"
--experiment="<Path_to_task_parameter_file.json>"```


##### Requirements

- Python v3.7.3
- numpy v1.16.2
- matplotlib v3.0.3
- scipy v1.2.1
- tqdm v4.40.2
- argparse v1.1
- json v2.0.9
