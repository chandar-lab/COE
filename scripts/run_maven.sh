#!/bin/bash

# MPE
envs=("mpe:SparseTag-v0")
pretraineds=("PretrainedTag")
# LBF
# envs=("lbforaging:Foraging-15x15-3p-5f-v2")
# SMAC
# envs=("3s_vs_5z")

algos=(noise_mix_episode)
lrs=(0.0001 0.0003 0.0005)
rnn_discrims=(0 1)
mi_intrins=(0 1)
int_reward_betas=(0 0.001 0.005 0.01 0.05 0.1 0.5)
noise_bandits=(0 1)
mixer=(qmix)
epsilon_starts=(0.0)
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
for discrim in ${rnn_discrims[@]}; do
for mi_intrin in ${mi_intrins[@]}; do
for int_beta in ${int_reward_betas[@]}; do
for nbandit in ${noise_bandits[@]}; do
for mxr in ${mixer[@]}; do
for eps_start in ${epsilon_starts[@]}; do
for seed in ${seeds[@]}; do


# mi_intrinsic and rnn_discrim are disjoint (src/learners/noise_q_learner.py)
if (( ${discrim} == 1 )) && (( ${mi_intrin} == 1 )); then
  continue
fi
if (( ${mi_intrin} == 1 )) && (( $(echo "${int_beta} == 0" |bc -l) )); then
  continue
fi
if (( ${discrim} == 1 )) && (( $(echo "${int_beta} > 0" |bc -l) )); then
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
hp_token=${algo}_${env_name}_lr${lr}_rd${discrim}_mi${mi_intrin}_ic${int_beta}_nb${nbandit}_${mxr}_ep${eps_start}_${seed}
temprunfilename="temprun_${hp_token}.sh"

echo "#!/bin/bash" >> ${temprunfilename}
echo "source ${HOME}/coe/bin/activate" >> ${temprunfilename}

tmp_config=tmp/${hp_token}
echo "python3 ${repo_dir}/make_config.py \
  -ic=${repo_dir}/src/config/algs/${algo}.yaml \
  -oc=${repo_dir}/src/config/algs/${tmp_config}.yaml \
  --local_results_path=${repo_dir}/results \
  --lr=${lr} \
  --rnn_discrim=${discrim} \
  --mi_intrinsic=${mi_intrin} \
  --mi_scaler=${int_beta} \
  --noise_bandit=${nbandit} \
  --mixer=${mxr} \
  --epsilon_start=${eps_start} \
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
