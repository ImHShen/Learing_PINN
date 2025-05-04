# PINN for 3D Radiative Heating of a Cube

## Description

This project implements a Physics-Informed Neural Network (PINN) using PyTorch to model and predict the 3D transient temperature field $u(x, y, z, t)$ inside a unit cube. The cube is initially at a uniform temperature and is heated by uniform thermal radiation from its surroundings.

The key features include:

* Solving the transient heat equation within the cube.
* Applying a non-linear radiative boundary condition on all faces.
* Parameterizing the model with the surrounding radiative temperature ($T_{surr}$) as an input, allowing prediction for different heating power levels with a single trained model.

## Project Structure

The project follows a standard machine learning project structure:

```
LEARNING_PINN
├── data/                     # Data files (raw, processed, external)
├── models/                   # Saved trained model files (e.g., .pth)
├── notebooks/                # Jupyter notebooks for exploration
├── reports/                  # Generated reports, figures, summaries
│   └── figures/              # Saved plots and visualizations
├── src/                      # Main source code directory
│   ├── __init__.py           # Makes 'src' a Python package
│   ├── config/               # Configuration related code/files
│   │   ├── __init__.py       # Makes 'src.config' a package
│   │   └── config.yaml     # <--- Configuration file MOVED HERE
│   ├── data/                 # Data loading/processing scripts (if needed)
│   │   └── __init__.py
│   ├── models/               # Model definition, training, evaluation scripts
│   │   ├── __init__.py
│   │   ├── model_definition.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── utils/                # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── helper_functions.py
│   │   └── pytorch_utils.py  # For PyTorch specific utils like device setting
│   └── visualization/        # Visualization scripts
│       ├── __init__.py
│       └── visualize.py
├── tests/                    # Unit tests
├── .gitignore                # Git ignore rules
├── README.md                 # Project description and setup guide
├── requirements.txt          # Project dependencies
└── main.py                   # Main script to execute the pipeline
```

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd ml-project-name
    ```

2. **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have a compatible PyTorch version installed for your hardware (CPU or CUDA GPU).*

## Configuration

All parameters for the simulation, model training, evaluation, and file paths are controlled via the main configuration file:

`src/config/config.yaml`

You can modify this file to change:

* Physical properties (`problem`, `material`, `constants` sections)
* Simulation settings (`T_max`, `T_surr` range)
* Neural network architecture (`training -> layers`)
* Training hyperparameters (`learning_rate`, `epochs`, `batch_sizes`)
* Evaluation settings (`evaluation -> T_surr_eval`, `grid` resolution)
* Output paths (`paths`)
* Execution flags (`run_training`, `run_evaluation`, etc.)

## Usage

To run the full pipeline (training, evaluation, visualization based on the config file), execute the main script from the project root directory:

```bash
python main.py --config src/config/config.yaml
```

* You can specify a different configuration file using the `--config` argument.
* The script will:
  * Load the specified configuration.
  * Optionally train the model and save it to the path specified in `paths -> model_save`.
  * Optionally evaluate the trained model for the `T_surr_eval` specified in the config.
  * Optionally generate and save plots (loss curve, temperature slices) to the paths specified in `paths`.

## Dependencies

The main dependencies are listed in `requirements.txt`. Key libraries include:

* PyTorch
* NumPy
* Matplotlib
* PyYAML

*(Optional dependencies for advanced visualization like PyVista are noted in `requirements.txt`)*

## Results

* Trained models are saved in the `models/` directory (or as configured).
* Generated plots (loss curves, temperature visualizations) are saved in the `reports/figures/` directory (or as configured).

*(Add any specific findings or example results here if desired)*

## TODO / Future Work

* Implement more sophisticated boundary conditions.
* Allow inference of unknown physical parameters (e.g., $k$, $\epsilon$).
* Explore different network architectures or activation functions.
* Add support for more complex geometries.
* Implement 3D visualization using PyVista.
* Add comprehensive unit tests.