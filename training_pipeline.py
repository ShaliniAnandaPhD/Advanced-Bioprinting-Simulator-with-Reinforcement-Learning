import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from organ_bioprint_env import OrganBioprintEnv
from rl_agent import BioprintingAgent
from reward_functions import composite_reward

def train_agent(env, agent, hyperparameters, save_path, log_dir, num_epochs=100, num_steps_per_epoch=1000, num_eval_episodes=5):
    """
    Train the reinforcement learning agent on the bioprinting environment.
    """
    # Create the PPO model
    model = PPO(policy=agent, env=env, **hyperparameters)

    # Create a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=num_steps_per_epoch, save_path=save_path, name_prefix='bioprinting_agent')

    # Create an evaluation callback to evaluate the model periodically
    eval_env = DummyVecEnv([lambda: OrganBioprintEnv()])
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path, log_path=log_dir, eval_freq=num_steps_per_epoch,
                                 deterministic=True, render=False, n_eval_episodes=num_eval_episodes)

    # Train the agent
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.learn(total_timesteps=num_steps_per_epoch, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)

    # Save the final trained model
    model.save(os.path.join(save_path, 'final_bioprinting_agent'))

    return model

def evaluate_agent(env, model, num_episodes=10):
    """
    Evaluate the trained agent on the bioprinting environment.
    """
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation results:")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Standard deviation of reward: {std_reward:.2f}")

if __name__ == '__main__':
    # Create the bioprinting environment
    env = OrganBioprintEnv()

    # Create the reinforcement learning agent
    agent = BioprintingAgent(observation_shape=env.observation_space.shape, action_space=env.action_space)

    # Define hyperparameters for training
    hyperparameters = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'n_steps': 2048,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'max_grad_norm': 0.5,
        'vf_coef': 0.5,
        'batch_size': 64,
        'verbose': 1
    }

    # Set up paths for saving models and logs
    save_path = 'trained_models'
    log_dir = 'training_logs'

    # Train the agent
    model = train_agent(env, agent, hyperparameters, save_path, log_dir)

    # Evaluate the trained agent
    evaluate_agent(env, model)