#!/bin/bash

# MPE
envs=("mpe:SparseTag-v0")
pretraineds=("PretrainedTag")
# LBF
# envs=("lbforaging:Foraging-15x15-3p-5f-v2")
# SMAC
# envs=("3s_vs_5z")

algos=(qmix_episode)
lrs=(0.0001 0.0003 0.0005)
mixer=(qmix)
epsilon_starts=(1.0)
epsilon_anneals=(50000 200000)
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
for mxr in ${mixer[@]}; do
for eps_start in ${epsilon_starts[@]}; do
for eps_anneal in ${epsilon_anneals[@]}; do
for seed in ${seeds[@]}; do


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
hp_token=${algo}_${env_name}_lr${lr}_${mxr}_ep${eps_start}_${eps_anneal}_${seed}
temprunfilename="temprun_${hp_token}.sh"

echo "#!/bin/bash" >> ${temprunfilename}
echo "source ${HOME}/coe/bin/activate" >> ${temprunfilename}

tmp_config=tmp/${hp_token}
echo "python3 ${repo_dir}/make_config.py \
  -ic=${repo_dir}/src/config/algs/${algo}.yaml \
  -oc=${repo_dir}/src/config/algs/${tmp_config}.yaml \
  --local_results_path=${repo_dir}/results \
  --lr=${lr} \
  --mixer=${mxr} \
  --epsilon_start=${eps_start} \
  --epsilon_anneal_time=${eps_anneal} \
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
