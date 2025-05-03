import argparse
import yaml # Using YAML for config is often cleaner
from src.models import train_model, evaluate_model
from src.visualization import visualize

def main(config_path):
    # Load configuration from YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded:")
        print(yaml.dump(config, indent=4))
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return

    # --- Training ---
    if config.get('run_training', True):
        print("\n--- Starting Training ---")
        training_history = train_model.train(config['training_params'])
        if config.get('plot_loss', True) and training_history:
             visualize.plot_loss_history(training_history, config.get('paths', {}).get('loss_curve_path', 'reports/figures/loss_curve.png'))
    else:
        print("\n--- Skipping Training ---")

    # --- Evaluation ---
    if config.get('run_evaluation', True):
        print("\n--- Starting Evaluation ---")
        model_path = config.get('training_params', {}).get('model_save_path', 'models/trained_pinn.pth')
        eval_params = config.get('evaluation_params', {})
        T_surr_to_eval = eval_params.get('T_surr_eval', 700.0) # Get specific T_surr from config

        # Load model
        layers = config.get('training_params', {}).get('layers', [5] + [128]*6 + [1])
        model = evaluate_model.PINN_Parametric(layers).to(evaluate_model.DEVICE)
        try:
             model.load_state_dict(torch.load(model_path, map_location=evaluate_model.DEVICE))
             print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model for evaluation: {e}")
            return

        # Create evaluation grid
        grid_conf = eval_params.get('grid', {'nx':15, 'ny':15, 'nz':15, 'nt':11})
        x_eval, y_eval, z_eval, t_eval, T_s_eval, grid_shape = evaluate_model.create_evaluation_grid(
            nx=grid_conf['nx'], ny=grid_conf['ny'], nz=grid_conf['nz'], nt=grid_conf['nt'],
            T_surr_eval=T_surr_to_eval
        )

        # Predict
        print(f"Predicting on grid for T_surr = {T_surr_to_eval} K...")
        u_predictions = evaluate_model.predict(model, x_eval, y_eval, z_eval, t_eval, T_s_eval)
        u_pred_grid = u_predictions.reshape(grid_shape)
        print("Prediction complete.")

        # --- Visualization ---
        if config.get('run_visualization', True):
            print("\n--- Starting Visualization ---")
            viz_params = config.get('visualization_params', {})
            slice_path = config.get('paths', {}).get('temp_slice_path', 'reports/figures/temp_slice_eval.png')
            # Add T_surr to filename maybe
            slice_path = slice_path.replace('.png', f'_Tsurr{int(T_surr_to_eval)}.png')

            visualize.plot_temperature_slice(
                u_pred_grid,
                grid_shape,
                slice_dim=viz_params.get('slice_dim', 'z'),
                slice_index=viz_params.get('slice_index', None), # None defaults to center
                time_index=viz_params.get('time_index', -1), # Default to final time
                T_surr_eval=T_surr_to_eval,
                save_path=slice_path
            )
        else:
            print("\n--- Skipping Visualization ---")

    else:
        print("\n--- Skipping Evaluation & Visualization ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PINN training and evaluation pipeline.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)