# A General Approach of Automated Environment Design for Learning the Optimal Power Flow

This is the accompanying repository to the publication **A General Approach of Automated Environment Design for Learning the Optimal Power Flow** 
by [Thomas Wolgast](https://orcid.org/0000-0002-9042-9964)
and [Astrid Nieße](https://orcid.org/0000-0003-1881-9172).

Note that most source code used for this paper can be found in 
https://gitlab.com/thomaswolgast/drl (RL algorithms) and 
https://github.com/Digitalized-Energy-Systems/opfgym 
(RL environments), 
which are work-in-progress and therefore continued in different repositories. 

# Installation
All experiments were performed with python 3.10. In an virtualenv, run `pip install -r requirements.txt` to install all dependencies in the right version at publication time (not the most recent). 

Note: [torch](https://pytorch.org/get-started/locally/) sometimes
needs to be installed manually before performing the previous step.

# Repository structure
- `run.sh`: A list of the commands performed to reproduce all experiments done for this publication. **Should not be run all at once!** Overall computation time will be multiple weeks. Use this file to copy-paste single commands from. Will automatically create a `data/` folder with the result files. 
- `LICENSE`: The license used for this work (MIT). 
- `requirements.txt`: Reference to the two previously mentioned repositories for simple installation. 
- `src/`: The source code to aggregate the results and create the figures of this exact publication. The source code for running the experiments is in the external repositories.
- `data/`: **[Optional folder that can be downloaded from Zenodo](https://zenodo.org/records/15315724)**. Data that was created by running the commands in `run.sh`. Contains hyperparameter information, environment design information, agent performances in the course of training, and the results of the hyperparameter optimization. Download this folder from Zenodo and place it in this repository to work with the data.

# Contact
For questions, feedback, or collaboration, contact the first author 
[Thomas Wolgast](https://orcid.org/0000-0002-9042-9964) (thomas.wolgast@uni-oldenburg.de). 