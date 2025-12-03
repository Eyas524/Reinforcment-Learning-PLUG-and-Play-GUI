#WORKING FINE/ NOT USED IN THIS GUI. For inspiring the solution
import gym

from PPOConfig import PPOConfig
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
#from stable_baselines3 import TRPO
from stable_baselines3 import SAC

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from Robodk_gui import RoboDKgui
import torch
from stable_baselines3.common.env_util import make_vec_env
from StopCallBack import EarlyStoppingCallback

class RobotTrain():
    def __init__(self):
        super(RobotTrain, self).__init__
        # Create the environment
        env = RoboDKgui()

        env = Monitor(env)  # Wrap the environment with Monitor

        # Check if the environment follows the Gymnasium interface
        print('Doing check environment')
        check_env(env)

        # Define evaluation environment and callback
        eval_env = RoboDKgui()
        eval_env = Monitor(eval_env)  # Wrap the evaluation environment with Monitor
        print('Doing eval callback')
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=5000, n_eval_episodes=5)

        early_stop = EarlyStoppingCallback(success_threshold=1)

        # Check if GPU is available and use it
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Model declaration is done")


        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device=device, 
            learning_rate=0.0005,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )



        # Train the agent with the callback
        print('Doing model learn')
        model.learn(total_timesteps=5000, callback=[eval_callback, early_stop])

        print('Doing model save')
        # Save the model
        model.save("ppo_ur10_robodk")

        # Test the trained model
        print('Doing model test')
        obs, _ = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if dones or truncated:
                print("Breaking the program")
                break

mytrain = RoboDKgui()


'''
stable_baselines3.ppo.PPO

(policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10,
 gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, 
 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, 
 rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, 
 tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)

'''