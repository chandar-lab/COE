# COE

## Note
This codebase accompanies paper [Conditionally Optimistic Exploration for Cooperative Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2303.09032) accepted at UAI 2023.

The codebase is based on the open-sourced framework [EPyMARL](https://github.com/uoe-agents/epymarl). Our implementations of [EMC](https://arxiv.org/abs/2111.11032) and [MAVEN](https://arxiv.org/abs/1910.07483) are based on the original [EMC code](https://github.com/kikojay/EMC) and [MAVEN code](https://github.com/AnujMahajanOxf/MAVEN), respectively.


## Run experiments
Please refer to the original README in EPyMARL for installation instructions.

To run an experiment with default configurations, run a command such as the following:
```shell
python3 src/main.py --config=ucb_mix_episode --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
where `--config` refers to the algorithm config file located in `src/config/algs`, and `--env-config` refers to the environment config file in `src/config/envs`.

To perform a hyperparameter search, either use the `search.py` script provided by authors of EPyMARL, or use scripts we provide in the `scripts` directory like:
```shell
bash scripts/run_ucb_mix.sh
```
We recommend our scripts as they enjoy more flexibility and clearity if experiments are scheduled using the SLURM manager.

## Results
We release a [colab notebook](https://colab.research.google.com/drive/1iRzQ1n2EXndUj4snkECJRXD4F-WiolyE) for the didactic multi-player game described in our paper.
Experiment results across MARL benchmarks are available [here](https://drive.google.com/file/d/1HDclfn5QFkKqVZN_sh7mifcJLnKdNv15/view?usp=sharing).