import optuna
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from utils.envs.balanced_env import BalancedStreetFighterEnv

def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        config = {
            'log_dir': './experiments/hpo/ppo/balanced/logs/',
            'frame_stack': 4,
            'total_timesteps': 100000,
            'check_freq': 10000,
            'checkpoint_dir': './experiments/hpo/ppo/balanced/checkpoints/'
        }
        env = BalancedStreetFighterEnv()
        env = Monitor(env, config['log_dir'])
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, config['frame_stack'], channels_order='last')

        model = PPO('CnnPolicy', env, tensorboard_log=config['log_dir'], verbose=0, **model_params)
        model.learn(total_timesteps=config['total_timesteps'])

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        save_path = os.path.join(config['checkpoint_dir'], f'trial_{trial.number}_best_model')
        model.save(save_path)

        return mean_reward

    except Exception as e:
        print(e)
        return -1000

if __name__ == "__main__":
    study_name = 'ppo_balanced'
    storage_name = f'sqlite:///{os.path.join("./experiments/hpo/ppo/balanced/", study_name)}.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
    study.optimize(optimize_agent, n_trials=100, n_jobs=1)
