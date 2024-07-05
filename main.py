import diff_rl
import os
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env

def main(
		env_id,
		algo,
		n_envs,
		iter_num,
		seed,
		learning_rate,
		n_sampled_actions):

	algo_name = algo
	log_name = algo_name
	algo = eval('diff_rl.'+ algo)
	env_kwargs = None

	env = make_vec_env(env_id=env_id, n_envs=n_envs, env_kwargs=env_kwargs)
	time = datetime.now().strftime("%Y_%m_%d-%H:%M")
	# make experiment directory
	logdir = f"{env_id}/logs/{env_id}+{log_name}+seed={seed}/{time}/"
	modeldir = f"{env_id}/models/{env_id}+{log_name}+seed={seed}/{time}/"

	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	model = algo(policy='MlpPolicy',
			  	env=env, 
				verbose=1, 
				tensorboard_log=logdir,
				seed=seed,
				learning_rate=learning_rate,
				n_sampled_actions=n_sampled_actions)

	for i in range(iter_num):
		model.learn(reset_num_timesteps=False, tb_log_name=f"{algo_name}")
		model.save(modeldir, f'{i * model.buffer_size}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Hopper-v4') 
    parser.add_argument('--algo', type=str, default='Diff_TD3') 
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--iter_num', type=int, default=10) # Total_timestep = iter_num * n_envs * n_steps, here is 2000 * 4 * 20480 = 1.2e7
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_sampled_actions', type=int, default=10)
    args = parser.parse_args()

    main(
	    args.env_id, 
		args.algo, 
		args.n_envs, 
		args.iter_num, 
		args.seed,
		args.learning_rate,
		args.n_sampled_actions)
