# From Event Logs to Decisions: History-Aware Reinforcement Learning for Process Optimization

## Summary of the Paper
Prescriptive process monitoring studies techniques that leverage event logs to recommend actions at runtime that improve process outcomes. A promising approach for next-action recommendations is reinforcement learning (RL). RL assumes that the decision-making process is Markovian, i.e., that the next observation depends only on the current observation and action, not on the history that led to it. However, business process executions are often subject to long-range  control-flow constraints that dictate which actions may occur, in what order, and under what conditions. Processes may also exhibit long-term dependencies between event attributes and actions. In this context, RL may produce suboptimal recommendations that violate the process control flow. The challenge is therefore to determine the minimum window of historical context needed to capture these dependencies without unnecessarily increasing the dimensionality of the observation space. To address these limitations, this paper proposes a framework, referred to as History-Aware RL, for learning history-dependent recommendations from event logs of non-Markovian processes. The approach employs statistical testing to determine the minimum window of historical context needed for decision-making and then trains an offline value-regularized deep RL algorithm on history-augmented observations. Our evaluation on synthetic and industrial event logs shows that the proposed method can produce recommendations that achieve better process outcomes and conformance than the baselines.

## Main Files Overview
#### `command.ls`
Contains all the python commands to replicate the results in the paper.

#### `src/`:
This directory contains all the Python scripts used for determining the Markov order of the decision-making process, training an offline CQL agent, and evaluating the policy.
- `MA_test.py`  
Applies the Markov Assumption statistical test by Shi et al. (2020) for `k = 1, ..., K`. 
- `CQL_offline.py`
Converts event data into k-step, history-augmented RL traces, trains an offline CQL agent to learn a history-dependent policy, and evaluates the learned policy using Fitted Q-Evaluation.
- `Evaluation.py`
Deploys the learned policy in the online setting of the synthetic SimBank process. 
- `MDP_functions.py`
Contains core functions for defining the Markov Decision Process, including state and terminal state definitions, reward computation, and removal of external actions from the agent's action space.

#### `dataset_manager/`:
This directory contains `DatasetManager.py` and `dataset_confs.py` (adapted from Teinemaa et al.), which define the column configuration of each event log (case ID, activity, timestamp, attributes, environmental actions) and handle loading, splitting, and prefix generation of the data. The event logs can be downloaded from [this link](<https://drive.google.com/drive/folders/1N9vkJ_Fd-MurLAC-aZC-fZzKl7SiexCI?usp=share_link>) and must be placed in a `data_logs/` folder at the root of the repository.

#### `SimBank/`:
This directory contains the SimBank implementation provided by De Moor et al.(2025).


#### `Final_plots_loss/`:
This directory contains plots of CQL agent training losses recorded throughout the training iterations.

