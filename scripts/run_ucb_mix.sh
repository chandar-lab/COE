#!/bin/bash

# MPE
envs=("mpe:SparseTag-v0")
pretraineds=("PretrainedTag")
# LBF
# envs=("lbforaging:Foraging-15x15-3p-5f-v2")
# SMAC
# envs=("3s_vs_5z")

algos=(ucb_mix_episode)
lrs=(0.0001 0.0003 0.0005)
macs=("ucb_cond_fac_mac") # ucb_mac, ucb_cond_fac_mac
counter="simhash_dict"
calculate_ucb_fns=("conditional") # "independent", "conditional"
key_dims=(8 12 16)
ucb_act_cps=(0 0.01 0.05)
ucb_learn_cps=(0 0.01 0.05)
ucb_learn_optim_learners=(0) # 0, 1
int_reward_betas=(0 0.01 0.05)
centralized_only=0  # (0,1), 1: only joint action bonus, int_reward_ind_beta=0
mixer=(qmix)
seeds=(0 1 2 3 4)
repo_dir="${HOME}/COE"

first_env=${envs[0]}
echo "fist env: ${first_env}"
env_config="gymma"
if [[ ${first_env} == "mpe"* ]]; then
  env_acr="mpe"
  time_limit="25"
elif [[ ${first_env} == "lbforaging"* ]]; then
  env_acr="lbforaging"
  time_limit="50"
else
  env_acr="smac"
  env_config="sc2"
  time_limit="50" # dummy
fi

mkdir -p ${repo_dir}/src/config/algs/tmp/ ${repo_dir}/results ${repo_dir}/logs/${env_acr}

for env_idx in ${!envs[@]}; do
for algo in ${algos[@]}; do
for lr in ${lrs[@]}; do
for mac in ${macs[@]}; do
for ucb_fn in ${calculate_ucb_fns[@]}; do
for key_dim in ${key_dims[@]}; do
for ucb_act_cp in ${ucb_act_cps[@]}; do
for ucb_learn_cp in ${ucb_learn_cps[@]}; do
for optim_learner in ${ucb_learn_optim_learners[@]}; do
for int_beta in ${int_reward_betas[@]}; do
for mxr in ${mixer[@]}; do
for seed in ${seeds[@]}; do

if (( $(echo "${ucb_act_cp} == 0" |bc -l) )) && (( $(echo "${ucb_learn_cp} == 0" |bc -l) )) && (( $(echo "${int_beta} == 0" |bc -l) )); then
  continue
fi

# set positive int_ind_beta if mac is ucb_cond_fac_mac
if [[ ${mac} == "ucb_cond_fac_mac" ]]; then
  int_ind_beta=${int_beta}
else
  int_ind_beta=0
fi

if (( ${centralized_only} == 1 )); then
  # want ucb_act_cp=ucb_learn_cp=0, int_beta>0
  if (( $(echo "${ucb_act_cp} > 0" |bc -l) )) || (( $(echo "${ucb_learn_cp} > 0" |bc -l) )) || (( $(echo "${int_beta} == 0" |bc -l) )); then
    continue
  fi
  int_ind_beta=0
fi

# ucb optim learner is applicable only when learn cp>0
if (( $(echo "${ucb_learn_cp} == 0" |bc -l) )) && (( ${optim_learner} == 1 )); then
  continue
fi
env=${envs[env_idx]}
# mpe wrapper
pretrain=${pretraineds[env_idx]}
if [[ ${env} == *"Tag"* ]] || [[ ${env} == *"Adversary"* ]]; then
  env_wrapper="env_args.pretrained_wrapper=${pretrain}"
else
  env_wrapper=""
fi
# env args
if [[ ${env_config} == "gymma" ]]; then
  env_args="env_args.time_limit=${time_limit} env_args.key=${env} ${env_wrapper}"
else
  env_args="env_args.map_name=${env}"
fi
env_name=${env//${env_acr}:/}
env_name=${env_name//-/_}
hp_token=${algo}_${env_name}_lr${lr}_${mac}_${counter}_${ucb_fn}_kd${key_dim}_ac${ucb_act_cp}_lc${ucb_learn_cp}_ol${optim_learner}_ic${int_beta}_${int_ind_beta}_${mxr}_${seed}
temprunfilename="temprun_${hp_token}.sh"

echo "#!/bin/bash" >> ${temprunfilename}
echo "source ${HOME}/coe/bin/activate" >> ${temprunfilename}

tmp_config=tmp/${hp_token}
echo "python3 ${repo_dir}/make_config.py \
  -ic=${repo_dir}/src/config/algs/${algo}.yaml \
  -oc=${repo_dir}/src/config/algs/${tmp_config}.yaml \
  --local_results_path=${repo_dir}/results \
  --lr=${lr} \
  --mac=${mac} \
  --calculate_ucb_fn=${ucb_fn} \
  --counter=${counter} \
  --key_dim=${key_dim} \
  --ucb_act_cp=${ucb_act_cp} \
  --ucb_learn_cp=${ucb_learn_cp} \
  --ucb_learn_optim_learner=${optim_learner} \
  --int_reward_ind_beta=${int_ind_beta} \
  --int_reward_cen_beta=${int_beta} \
  --mixer=${mxr} \
  --seed=${seed}" >> ${temprunfilename}
echo "python3 src/main.py \
  --config=${tmp_config} \
  --env-config=${env_config} \
  with ${env_args}" >> ${temprunfilename}

eval "bash ${temprunfilename}"
rm ${temprunfilename}

done
done
done
done
done
done
done
done
done
done
done
done
