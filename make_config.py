import argparse
import yaml
import os

bool_vars = [
    "standardise_rewards",
    "double_q",
    "use_rnn",
    "ucb_learn_optim_learner",
    "ucb_optim_init",
    "ucb_conf_decay",
    "load_ckpt",
    "rnn_discrim",
    "noise_bandit",
    "mi_intrinsic",
]

def make_changes(args, input_config):
    for key, val in vars(args).items():
        if key in ["input_config",]:
            continue
        elif key == "output_config":
            hp_token = val.split("/")[-1][:-5]
            input_config["hp_token"] = hp_token
        elif key in bool_vars:
            input_config[key] = bool(val)
        elif key == "target_update_interval_or_tau":
            input_config[key] = val if val < 1.0 else int(val)
        else:
            input_config[key] = val

    if input_config["load_ckpt"]:
        input_config["checkpoint_path"] = os.path.join(
            input_config["local_results_path"],
            "models",
            input_config["hp_token"],
        )

    return input_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-ic', '--input-config')
    parser.add_argument('-oc', '--output-config')
    parser.add_argument('--local_results_path', type=str, help='output dir')

    parser.add_argument('--action_selector', type=str)
    parser.add_argument('--epsilon_start', type=float)
    parser.add_argument('--epsilon_finish', type=float)
    parser.add_argument('--epsilon_anneal_time', type=int)
    parser.add_argument('--evaluation_epsilon', type=float)

    parser.add_argument('--runner', type=str)
    parser.add_argument('--batch_size_run', type=int)

    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--target_update_interval_or_tau', type=float)

    parser.add_argument('--mac', type=str)
    parser.add_argument('--agent', type=str)

    parser.add_argument("--obs_agent_id", action="store_true")
    parser.add_argument("--obs_last_action", action="store_true")
    parser.add_argument("--obs_individual_obs", action="store_true")

    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument("--standardise_rewards", type=int, default=1, help="whether to standardize rewards in batch")
    parser.add_argument('--agent_output_type', type=str)
    parser.add_argument('--learner', type=str)
    parser.add_argument("--double_q", type=int, default=1, help="whether to perform double q learning")
    parser.add_argument('--mixer', type=str)
    parser.add_argument("--use_rnn", type=int, default=1, help="whether to use rnn")
    parser.add_argument('--mixing_embed_dim', type=int)
    parser.add_argument('--hypernet_layers', type=int)
    parser.add_argument('--hypernet_embed', type=int)

    parser.add_argument('--name', type=str)

    parser.add_argument('--t_max', type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--save_model_interval', type=int)
    parser.add_argument('--test_nepisode', type=int)
    parser.add_argument('--test_interval', type=int)
    parser.add_argument('--log_interval', type=int)

    parser.add_argument("--load_ckpt", type=int, default=0, help="whether to load from checkpoint")
    parser.add_argument("--counter", type=str, default="simhash",
        choices=["simhash", "simhash_dict",],
        help="counter model")
    parser.add_argument('--decay_factor', type=float, default=1.0, help='incremented count')
    parser.add_argument('--key_dim', type=int, default=16, help='simhash key dimension')
    parser.add_argument('--ucb_act_cp', type=float, default=1.0, help='scale of ucb bonus')
    parser.add_argument('--ucb_learn_cp', type=float, default=1.0, help='scale of ucb bonus during learning')
    parser.add_argument('--ucb_learn_optim_learner', type=int, default=1, help='whether to add bonus to learner net')
    parser.add_argument('--int_reward_ind_beta', type=float, default=0.0, help='scale of intrinsic reward added to individual agent')
    parser.add_argument('--int_reward_cen_beta', type=float, default=0.0, help='scale of intrinsic reward added to centralized agent')
    parser.add_argument('--int_reward_clip', type=float, default=5.0, help='maximum magnitude of intrinsic reward')
    parser.add_argument("--confidence_fn", type=str, default="ucb1",
        choices=["ucb1", "ucb_asymptotic", "ucb_modified", "modified_uct"],
        help="confidence function")
    parser.add_argument("--calculate_ucb_fn", type=str, default="conditional", choices=["conditional", "independent"], help="agent ucb function")
    parser.add_argument('--ucb_optim_init', type=int, default=0, help='ucb optimistic initial confidence')
    parser.add_argument('--ucb_conf_decay', type=int, default=1, help='whether to decay confidence wrt depth in tree')

    parser.add_argument('--rnn_discrim', type=int, default=0, help='whether to use rnn discriminator')
    parser.add_argument('--noise_bandit', type=int, default=0, help='whether to use trainable noise')
    parser.add_argument('--mi_intrinsic', type=int, default=0, help='whether to use intrinsic reward')
    parser.add_argument('--mi_scaler', type=float, default=0.0, help='scale of intrinsic reward')

    parser.add_argument('--curiosity_scale', type=float, default=0.0, help='scale of intrinsic reward')

    parser.add_argument('--seed', type=int, help='Random Seed to use in the experiment')

    args = parser.parse_args()

    with open(args.input_config) as f:
        input_config = yaml.safe_load(f)

    input_config = make_changes(args, input_config)

    with open(args.output_config, 'w+') as f:
        yaml.safe_dump(input_config, stream=f, default_flow_style=False)
