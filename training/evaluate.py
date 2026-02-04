"""
Simple evaluation script that saves agent states to CSV for analysis.

Usage:
    python evaluate_simple.py --checkpoint path/to/checkpoint --num-episodes 5
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy

from envs.energy_env import P2PEnergyEnv
from training.policy_mapping import energy_policy_mapping_fn


def find_latest_checkpoint(experiment_dir="./exp_results/energy_market_training"):
    """Find the latest checkpoint in the experiment directory."""
    # Convert to absolute path
    experiment_dir = os.path.abspath(experiment_dir)
    trial_dirs = list(Path(experiment_dir).glob("PPO_*"))
    if not trial_dirs:
        raise ValueError(f"No PPO directories found in {experiment_dir}")
    
    latest_trial = max(trial_dirs, key=lambda p: p.stat().st_mtime)
    checkpoints = list(latest_trial.glob("checkpoint_*"))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {latest_trial}")
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('_')[-1]))
    # Return absolute path
    return os.path.abspath(str(latest_checkpoint))


def load_rlmodules(checkpoint_path):
    """Load RLModules for each policy from checkpoint."""
    base_path = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    
    rlmodules = {}
    
    # Load seller_policy and buyer_policy
    for policy in ["seller_policy", "buyer_policy"]:
        policy_path = base_path / policy
        if policy_path.exists():
            print(f"  Loading {policy} from {policy_path}")
            rlmodules[policy] = RLModule.from_checkpoint(str(policy_path))
            rlmodules[policy].eval()  # Set to evaluation mode (no training)
        else:
            raise ValueError(f"Could not find RLModule for {policy} at {policy_path}")
    
    return rlmodules


def evaluate_and_save_states(checkpoint_path, num_episodes=5, output_file=None):
    """
    Run trained policies and save all agent states to CSV.
    
    Args:
        checkpoint_path: Path to checkpoint
        num_episodes: Number of episodes to run
        output_file: CSV filename to save results (if None, saves in checkpoint folder)
    """
    
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # If no output file specified, create one in the checkpoint's parent directory
    if output_file is None:
        checkpoint_dir = Path(checkpoint_path).parent
        output_file = checkpoint_dir / "evaluation_agent_states.csv"
    else:
        output_file = Path(output_file)
    
    print(f"Output will be saved to: {output_file}\n")
    
    # Load trained RLModules
    print("Loading trained policies...")
    rlmodules = load_rlmodules(checkpoint_path)
    print("Policies loaded!\n")
    
    # Create environment
    print("Creating environment...")
    env = P2PEnergyEnv()
    print("Environment created!\n")
    
    # Storage for all states across episodes
    all_data = []
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}...")
        
        obs, info = env.reset()
        done = {"__all__": False}
        step = 0
        
        while not done["__all__"]:
            # Compute actions for all agents
            actions = {}
            
            for agent_id, agent_obs in obs.items():
                # Determine which policy to use
                policy_id = energy_policy_mapping_fn(agent_id, episode, None)
                
                # Get the corresponding RLModule
                rl_module = rlmodules[policy_id]
                
                # Prepare input (add batch dimension)
                input_dict = {
                    Columns.OBS: torch.from_numpy(agent_obs).unsqueeze(0).float()
                }
                
                # Forward pass (inference mode, no exploration)
                with torch.no_grad():
                    rl_module_out = rl_module.forward_inference(input_dict)

                dist_cls = rl_module.get_inference_action_dist_cls()

                action_dist = dist_cls.from_logits(rl_module_out[Columns.ACTION_DIST_INPUTS])

                action_det = action_dist._dist.mean

                action_np = convert_to_numpy(action_det)[0]

                space = env.action_spaces[agent_id]
                
                action_np = np.clip(action_np, space.low, space.high)
                # Extract action from output
                # For continuous action spaces (Box), we get mean and log_std
                # We only want the mean (deterministic action)
                # action_dist_inputs = rl_module_out[Columns.ACTION_DIST_INPUTS]
                
                # Convert to numpy
                # action_np = convert_to_numpy(action_dist_inputs).squeeze(0)
                # action_np = convert_to_numpy(rl_module_out[Columns.ACTIONS])[0]
                # print(action_np)
                # For Box spaces with Gaussian distribution, the output is [mean, log_std]
                # We only need the mean (first half)
                if policy_id == "seller_policy":
                    # Seller action space is Box(shape=(2,))
                    # Output is shape (4,) = [mean_0, mean_1, log_std_0, log_std_1]
                    action = action_np[:2]  # Take only the means
                elif policy_id == "buyer_policy":
                    # Buyer action space is Box(shape=(1,))
                    # Output is shape (2,) = [mean, log_std]
                    action = action_np[:1]  # Take only the mean
                else:
                    action = action_np
                actions[agent_id] = action
            
            # Step environment
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Record data for this timestep
            step_data = {
                'episode': episode + 1,
                'step': step,
            }
            
            # Get seller states (power allocations)
            for seller in env.sellers:
                agent_id = seller.group_name
                # seller.state is array of power allocations [P_i,0, P_i,1]
                step_data[f'{agent_id}_power_to_buyer0'] = seller.state[0]
                step_data[f'{agent_id}_power_to_buyer1'] = seller.state[1]
                step_data[f'{agent_id}_total_power'] = np.sum(seller.state)
                step_data[f'{agent_id}_capacity'] = seller.net[0]  # Available capacity
                step_data[f'{agent_id}_reward'] = rewards.get(agent_id, 0.0)
            
            # Get buyer states (prices)
            for buyer in env.buyers:
                agent_id = buyer.group_name
                # buyer.state is a single price value
                step_data[f'{agent_id}_price'] = float(buyer.state)
                step_data[f'{agent_id}_demand'] = buyer.net[0]  # Demand
                step_data[f'{agent_id}_reward'] = rewards.get(agent_id, 0.0)
            
            # Add Lagrange multipliers if you want to track them
            step_data['lambda_0'] = env.lagrange[0]
            step_data['lambda_1'] = env.lagrange[1]
            step_data['lambda_2'] = env.lagrange[2]
            step_data['mu'] = env.lagrange[3]
            
            all_data.append(step_data)
            
            # Update for next iteration
            obs = next_obs
            done = terminateds
            step += 1
        
        print(f"  Completed in {step} steps\n")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(str(output_file), index=False)
    
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}\n")
    
    # Show preview
    print("Preview of data:")
    print(df.head(10))
    print("\nBasic statistics:")
    print(df.describe())
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate and save agent states to CSV")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (if not provided, will find latest)"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 5)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (default: saves as 'evaluation_agent_states.csv' in checkpoint folder)"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint if not provided
    if args.checkpoint is None:
        print("No checkpoint provided, searching for latest...")
        checkpoint_path = find_latest_checkpoint()
    else:
        # Convert to absolute path
        checkpoint_path = os.path.abspath(args.checkpoint)
    
    # Run evaluation
    df = evaluate_and_save_states(
        checkpoint_path=checkpoint_path,
        num_episodes=args.num_episodes,
        output_file=args.output
    )
    
    if args.output:
        print(f"\n✓ Evaluation complete! Check '{args.output}' for results.")
    else:
        print(f"\n✓ Evaluation complete! Check the checkpoint folder for 'evaluation_agent_states.csv'.")


if __name__ == "__main__":
    main()