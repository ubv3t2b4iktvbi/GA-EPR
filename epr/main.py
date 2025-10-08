import torch
import os
import numpy as np
from utils import ModelArgs, Problem, set_seed_everywhere
from network import NetworkWrapper
from landscape import EnergyLandscape
from dynamics import get_force_

def main(config_path=None):
    """Main entry point for training and visualization"""
    # Initialize system arguments from YAML config or defaults
    if config_path and os.path.exists(config_path):
        args = ModelArgs.from_yaml(config_path)
        # Also load problem config if it exists in the same directory
        problem_config_path = os.path.join(os.path.dirname(config_path), 'problem.yaml')
        if os.path.exists(problem_config_path):
            problem = Problem.from_yaml(problem_config_path)
        else:
            problem = Problem()
    else:
        # Fallback to default configuration
        args = ModelArgs()
        if config_path:
            args.prefix = os.path.dirname(os.path.abspath(config_path))
        else:
            # 设置默认前缀路径
            default_config_dir = os.path.join("results", "bistable")
            args.prefix = os.path.abspath(default_config_dir)

        args.viz_interval = 20  # Visualize every 10 epochs
        args.save_interval = 50  # Save checkpoint every 50 epochs
        args.num_epochs = 1000  # Total training epochs
        args.val_interval = 20  # Validate every 5 epochs (new parameter)
        
        # 添加力场类型和参数配置
        # 根据config_path设置force_type和相应的force_params，避免冗余
        if config_path and 'sl' in config_path.lower():
            args.force_type = 'Stuart_Landau'
            args.force_params = {'lambda_': 2.5, 'm1': -1.5, 'm2': 1.5}
        elif config_path and 'biochemical' in config_path.lower():
            args.force_type = 'Biochemical'
            args.force_params = {}
        elif config_path and 'bistable' in config_path.lower():
            args.force_type = 'bistable'
            args.force_params = {'a': 1.0}
        else:
            args.force_type = 'bistable'  # 默认值
            args.force_params = {'a': 1.0}
        
        # Initialize problem definition
        problem = Problem()
        problem.noise_strength = 0.01  # Diffusion strength
        problem.dnn_sample_size = 10000  # Samples for DNN training
        problem.flow_sample_size = 0  # Samples for flow MLE training
        # problem.flow_constraint_sample_size = 5000  # Samples for flow constraints
    
    # Create output directories
    os.makedirs(args.prefix, exist_ok=True)
    # Save experiment parameters to YAML files
    args.save_to_yaml(os.path.join(args.prefix, 'config.yaml'))
    problem.save_to_yaml(os.path.join(args.prefix, 'problem.yaml'))

    # Set random seed for reproducibility
    set_seed_everywhere(42)
    
    # Initialize neural networks
    network = NetworkWrapper(args, problem)
    
    # Create force function using factory method with configured parameters
    force_params = args.force_params.get(args.force_type, {}) if isinstance(args.force_params, dict) else {}
    force_fn = get_force_(args.force_type, **force_params)

    # Initialize energy landscape trainer
    landscape = EnergyLandscape(args, problem, network, force_fn)
    
    # Start training
    print("Starting training...")
    landscape.train()
    print("Training complete.")
    
    # Generate final visualization
    print("Generating final visualizations...")
    landscape._visualize(args.num_epochs)
    print(f"Results saved to {args.prefix}")

if __name__ == "__main__":
    import sys
    import os
    
    print("🚀 Starting Ga-EPR simulation...")
    default_config_dir = os.path.join("results", "sl")
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            # If input is a directory, look for config.yaml in that directory
            config_path = os.path.join(input_path, 'config.yaml')
            problem_path = os.path.join(input_path, 'problem.yaml')
            if not os.path.exists(config_path):
                print(f"\n❌ Error: config.yaml not found in directory {input_path}")
                print(f"   Please ensure config.yaml exists in the specified directory.")
                print(f"   Usage: python main.py [config_directory | config_file]")
                print(f"   Example: python main.py {default_config_dir}")
                sys.exit(1)
        else:
            config_path = input_path
            problem_path = os.path.join(os.path.dirname(input_path), 'problem.yaml') if os.path.dirname(input_path) else 'problem.yaml'
            if not os.path.exists(config_path):
                print(f"\n❌ Error: Config file {config_path} does not exist")
                print(f"   Usage: python main.py [config_directory | config_file]")
                print(f"   Example: python main.py {default_config_dir}")
                sys.exit(1)
    else:
        os.makedirs(default_config_dir, exist_ok=True)
        config_path = os.path.join(default_config_dir, 'config.yaml')
        problem_path = os.path.join(default_config_dir, 'problem.yaml')
        print(f"\n⚠️  Using default configuration. Config file {config_path} may not exist, but will use default settings")
    
    main(config_path)
    print("\n✅ Simulation completed successfully!")