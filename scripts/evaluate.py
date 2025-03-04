import argparse
import gymnasium as gym
import torch
import numpy as np

def evaluate(env_name, model_path, num_episodes=100):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    q_network = torch.load(model_path)
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    evaluate(args.env, args.model, args.episodes)