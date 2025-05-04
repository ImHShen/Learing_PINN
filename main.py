import argparse
import yaml
import torch
import os
import sys

# Ensure the src directory is in the Python path
# This allows absolute imports like 'from src.models import ...'
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now perform imports using absolute paths from src
from src.models import train_model, evaluate_model
from src.visualization import visualize
from src.utils.pytorch_utils import set_seed # Optional seeding

def main(config_path_relative):
    # Construct absolute path to config file relative to main.py's location
    config_path_abs = os.path.join(project_root, config_path_relative)

    # Load configuration from YAML file
    try:
        with open(config_path_abs, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path_abs}")
        # print(yaml.dump(config, indent=4)) # Optionally print loaded config
    except FileNotFoundError:
         print(f"Error: Configuration file not found at {config_path_abs}")
         sys.exit(1)
    except Exception as e:
        print(f"Error loading config file {config_path_abs}: {e}")
        sys.exit(1)

    # Optional: Set random seed for reproducibility using value from config
    if 'global_seed' in config:
        set_seed(config['global_seed'])
        print(f"Random seed set to {config['global_seed']}")

    # --- Training ---
    if config.get('run_training', True):
        print("\n--- Starting Training ---")
        # Pass the whole config dictionary to train function
        training_history = train_model.train(config)

        if config.get('plot_loss', True) and training_history:
             loss_path_relative = config.get('paths', {}).get('loss_curve', 'reports/figures/loss_curve.png')
             loss_path_abs = os.path.join(project_root, loss_path_relative)
             visualize.plot_loss_history(training_history, loss_path_abs)
    else:
        print("\n--- Skipping Training ---")

    # --- Evaluation & Visualization ---
    if config.get('run_evaluation', True):
        print("\n--- Starting Evaluation ---")
        # Ensure model path is relative to project root
        model_path_relative = config.get('paths', {}).get('model_save', 'models/trained_pinn.pth')
        model_path_abs = os.path.join(project_root, model_path_relative)

        eval_params = config.get('evaluation', {})
        train_params = config.get('training', {}) # Need layers from training config
        T_surr_to_eval = eval_params.get('T_surr_eval', 800.0)

        # Load model (needs model_definition and device)
        from src.models.model_definition import PINN_Parametric
        from src.utils.pytorch_utils import DEVICE # Import device for evaluation
        layers = train_params.get('layers', [5] + [128]*6 + [1]) # Default if not in config
        model = PINN_Parametric(layers).to(DEVICE)

        try:
             # Load state dict checking if file exists
             if not os.path.exists(model_path_abs):
                  raise FileNotFoundError(f"Model file not found at {model_path_abs}. Cannot evaluate.")
             model.load_state_dict(torch.load(model_path_abs, map_location=DEVICE))
             print(f"Model loaded from {model_path_abs}")
        except Exception as e:
            print(f"Failed to load model for evaluation: {e}")
            return # Exit if model can't be loaded

        # Create evaluation grid (needs problem constants from config)
        problem_cfg = config.get('problem', {})
        L_val = problem_cfg.get('L', 1.0)
        T_max_val = problem_cfg.get('T_max', 1000.0)

        grid_conf = eval_params.get('grid', {'nx':15, 'ny':15, 'nz':15, 'nt':11})
        x_eval, y_eval, z_eval, t_eval, T_s_eval, grid_shape = evaluate_model.create_evaluation_grid(
            nx=grid_conf['nx'], ny=grid_conf['ny'], nz=grid_conf['nz'], nt=grid_conf['nt'],
            T_surr_eval=T_surr_to_eval,
            L_val=L_val, # Pass L and T_max needed for grid creation
            T_max_val=T_max_val
        )

        # Predict
        print(f"Predicting on grid for T_surr = {T_surr_to_eval} K...")
        u_predictions = evaluate_model.predict(model, x_eval, y_eval, z_eval, t_eval, T_s_eval)
        u_pred_grid = u_predictions.reshape(grid_shape)
        print("Prediction complete.")

        # --- Visualization ---
        if config.get('run_visualization', True):
            print("\n--- Starting Visualization ---")
            viz_params = config.get('visualization', {})
            paths_cfg = config.get('paths', {})
            slice_base_relative = paths_cfg.get('temp_slice_base', 'reports/figures/temperature_slice')
            slice_base_abs = os.path.join(project_root, slice_base_relative)
            # Add T_surr to filename
            slice_path_abs = slice_base_abs + f'_Tsurr{int(T_surr_to_eval)}.png'

            visualize.plot_temperature_slice(
                u_pred_grid,
                grid_shape,
                slice_dim=viz_params.get('slice_dim', 'z'),
                slice_index=viz_params.get('slice_index', None),
                time_index=viz_params.get('time_index', -1),
                T_surr_eval=T_surr_to_eval,
                save_path=slice_path_abs,
                L_val=L_val, # Pass plotting bounds if needed
                T_max_val=T_max_val
            )
        else:
            print("\n--- Skipping Visualization ---")
    else:
        print("\n--- Skipping Evaluation & Visualization ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PINN training and evaluation pipeline.')
    # Default path now points inside src/config/
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file (relative to project root).')
    args = parser.parse_args()
    main(args.config)