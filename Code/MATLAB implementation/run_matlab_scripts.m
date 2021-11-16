%%%%-------------------------------------------------------------------------%%%%

%%% Simulates all the task variants using the MATLAB codes provided by the authors

%%%%-------------------------------------------------------------------------%%%%


close all
clear all

set(groot,'defaultFigureVisible','off')                                                         % suppresses display of figures

res_folder =  "Results/";


%  Sets the number of timesteps in each trial
triallen = 1e4;

% Set the colour codes
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
orange      = [1 0.5 0];
purple      = [.2 0.1 .9];
grey        = [0.2 0.2 0.2];

% Set the seed for the random generator for the overall run (Note: The seeds 0 and 5489 are equivalent in MATLAB.)
rng(0);

%%%--------------------------------%%%
%% Simulate all tasks using the default seed 5489.
%%%--------------------------------%%%

% Simulate FORCE algorithm on Task 1
Task1_FORCE(triallen, res_folder+'FORCE_Task1/', green, red, orange, purple, grey, 5489);
close all;
clear Task1_FORCE;

% Simulate RMHL algorithm on Task 1
Task1_RMHL(triallen, res_folder+'RMHL_Task1/', green, red, orange, purple, grey, 5489);
close all;
clear Task1_RMHL;

% Simulate SUPERTREX algorithm on Task 1
Task1_ST(triallen, res_folder+'ST_Task1/', green, red, orange, purple, grey, 5489);
close all;
clear Task1_ST;


% Simulate RMHL algorithm on the basic Task 2 with 2 segments of equal length
Task2_RMHL_Seg2(triallen, res_folder+'RMHL_Task2_Seg2/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_RMHL_Seg2;

% Simulate RMHL algorithm on a Task 2 variant with 2 segments  with varying lengths
Task2_RMHL_Seg2_Var(triallen, res_folder+'RMHL_Task2_Seg2_Var/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_RMHL_Seg2_Var;

% Simulate RMHL algorithm on a Task 2 variant with 3 segments  with equal lengths
Task2_RMHL_Seg3(triallen, res_folder+'RMHL_Task2_Seg3/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_RMHL_Seg3;

% Simulate RMHL algorithm on a Task 2 variant with 3 segments  with varying lengths
Task2_RMHL_Seg3_Var(triallen, res_folder+'RMHL_Task2_Seg3_Var/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_RMHL_Seg3_Var;

% Simulate SUPERTREX algorithm on the basic Task 2 with 2 segments  with equal lengths
Task2_ST_Seg2(triallen, res_folder+'ST_Task2_Seg2/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_ST_Seg2;

% Simulate SUPERTREX algorithm on a Task 2 variant with 2 segments  with varying lengths
Task2_ST_Seg2_Var(triallen, res_folder+'ST_Task2_Seg2_Var/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_ST_Seg2_Var;

% Simulate SUPERTREX algorithm on a Task 2 variant with 3 segments  with equal lengths
Task2_ST_Seg3(triallen, res_folder+'ST_Task2_Seg3/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_ST_Seg3;

% Simulate SUPERTREX algorithm on a Task 2 variant with 3 segments  with varying lengths
Task2_ST_Seg3_Var(triallen, res_folder+'ST_Task2_Seg3_Var/', green, red, orange, purple, grey, 5489);
close all;
clear Task2_ST_Seg3_Var;


% Simulate RMHL algorithm on the basic Task 3 variant with different costs
Task3_RMHL(triallen, res_folder+'RMHL_Task3/', green, red, orange, purple, grey, 5489);
close all;
clear Task3_RMHL;

% Simulate RMHL algorithm on a Task 3 variant with different cost
Task3_RMHL_beta(triallen, res_folder+'RMHL_Task3_01/', green, red, orange, purple, grey, 5489);
close all;
clear Task3_RMHL;

% Simulate SUPERTREX algorithm on the basic Task 3 variant with different costs
Task3_ST(triallen, res_folder+'ST_Task3/', green, red, orange, purple, grey, 5489);
close all;
clear Task3_ST;

% Simulate SUPERTREX algorithm on a Task 3 variant with different cost
Task3_ST_beta(triallen, res_folder+'ST_Task3_01/', green, red, orange, purple, grey, 5489);
close all;
clear Task3_ST;

%%%--------------------------------%%%
%% Simulate all tasks using different seeds for the random generator
%%%--------------------------------%%%

for i = 1:10

    % Simulate FORCE algorithm on Task 1
    Task1_FORCE(triallen, res_folder+'FORCE_Task1/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task1_FORCE;

    % Simulate RMHL algorithm on Task 1
    Task1_RMHL(triallen, res_folder+'RMHL_Task1/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task1_RMHL;

    % Simulate SUPERTREX algorithm on Task 1
    Task1_ST(triallen, res_folder+'ST_Task1/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task1_ST;


    % Simulate RMHL algorithm on the basic Task 2 with 2 segments of equal length
    Task2_RMHL_Seg2(triallen, res_folder+'RMHL_Task2_Seg2/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_RMHL_Seg2;

    % Simulate RMHL algorithm on a Task 2 variant with 2 segments  with varying lengths
    Task2_RMHL_Seg2_Var(triallen, res_folder+'RMHL_Task2_Seg2_Var/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_RMHL_Seg2_Var;

    % Simulate RMHL algorithm on a Task 2 variant with 3 segments  with equal lengths
    Task2_RMHL_Seg3(triallen, res_folder+'RMHL_Task2_Seg3/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_RMHL_Seg3;

    % Simulate RMHL algorithm on a Task 2 variant with 3 segments  with varying lengths
    Task2_RMHL_Seg3_Var(triallen, res_folder+'RMHL_Task2_Seg3_Var/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_RMHL_Seg3_Var;

    % Simulate SUPERTREX algorithm on the basic Task 2 with 2 segments  with equal lengths
    Task2_ST_Seg2(triallen, res_folder+'ST_Task2_Seg2/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_ST_Seg2;

    % Simulate SUPERTREX algorithm on a Task 2 variant with 2 segments  with varying lengths
    Task2_ST_Seg2_Var(triallen, res_folder+'ST_Task2_Seg2_Var/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_ST_Seg2_Var;

    % Simulate SUPERTREX algorithm on a Task 2 variant with 3 segments  with equal lengths
    Task2_ST_Seg3(triallen, res_folder+'ST_Task2_Seg3/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_ST_Seg3;

    % Simulate SUPERTREX algorithm on a Task 2 variant with 3 segments  with varying lengths
    Task2_ST_Seg3_Var(triallen, res_folder+'ST_Task2_Seg3_Var/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task2_ST_Seg3_Var;


    % Simulate RMHL algorithm on the basic Task 3 variant with different costs
    Task3_RMHL(triallen, res_folder+'RMHL_Task3/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task3_RMHL;

    % Simulate RMHL algorithm on a Task 3 variant with different cost
    Task3_RMHL_beta(triallen, res_folder+'RMHL_Task3_01/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task3_RMHL;

    % Simulate SUPERTREX algorithm on the basic Task 3 variant with different costs
    Task3_ST(triallen, res_folder+'ST_Task3/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task3_ST;

    % Simulate SUPERTREX algorithm on a Task 3 variant with different cost
    Task3_ST_beta(triallen, res_folder+'ST_Task3_01/', green, red, orange, purple, grey, randi(1e9));
    close all;
    clear Task3_ST;

end
