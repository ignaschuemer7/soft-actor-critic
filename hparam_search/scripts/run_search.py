import argparse
import yaml
import subprocess
import optuna
import os
import shutil


def objective(trial, args, search_dir):
    # Load the base config
    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    # Load the search space
    with open(args.search_config, "r") as f:
        search_config = yaml.safe_load(f)["search_space"]

    # Create a trial-specific config by deep copying the base config
    trial_config = base_config.copy()
    for key, value in base_config.items():
        trial_config[key] = value.copy() if isinstance(value, dict) else value

    # Suggest hyperparameters
    for section, params in search_config.items():
        if section not in trial_config:
            trial_config[section] = {}
        for param, settings in params.items():
            if settings["type"] == "categorical":
                trial_config[section][param] = trial.suggest_categorical(
                    f"{section}.{param}", settings["choices"]
                )
            elif settings["type"] == "uniform":
                trial_config[section][param] = trial.suggest_float(
                    f"{section}.{param}", settings["low"], settings["high"]
                )
            elif settings["type"] == "loguniform":
                trial_config[section][param] = trial.suggest_float(
                    f"{section}.{param}", settings["low"], settings["high"], log=True
                )

    # Create a unique directory for this trial's run
    trial_run_dir = os.path.join(search_dir, f"trial_{trial.number}")
    os.makedirs(trial_run_dir, exist_ok=True)

    # Set the experiment name in the logger config to include the trial number
    trial_config["logger"][
        "experiment_name"
    ] = f"sac-hparam-search-trial-{trial.number}"
    trial_config["logger"]["log_dir"] = trial_run_dir

    # Path for the trial's config file
    trial_config_path = os.path.join(trial_run_dir, "config.yaml")

    with open(trial_config_path, "w") as f:
        yaml.dump(trial_config, f)

    # Run main.py with the trial's config
    command = ["python", "main.py", "--config", trial_config_path]

    print(f"--- Starting Trial {trial.number} ---")
    print(f"Config path: {trial_config_path}")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Save stdout and stderr to files
        with open(os.path.join(trial_run_dir, "stdout.log"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(trial_run_dir, "stderr.log"), "w") as f:
            f.write(result.stderr)

        # Find the final average return in the output
        output_lines = result.stdout.strip().split("\n")
        for line in reversed(output_lines):
            if "Final average return:" in line:
                final_return = float(line.split(":")[1].strip())
                print(f"--- Trial {trial.number} Finished ---")
                print(f"Final average return: {final_return}")
                return final_return

        print(
            f"Warning: Could not find 'Final average return:' in the output for trial {trial.number}."
        )
        raise optuna.exceptions.TrialPruned("Could not parse final return.")

    except subprocess.CalledProcessError as e:
        print(f"Error running trial {trial.number}:")
        print(e.stdout)
        print(e.stderr)
        # Save stdout and stderr to files even if it fails
        with open(os.path.join(trial_run_dir, "stdout.log"), "w") as f:
            f.write(e.stdout)
        with open(os.path.join(trial_run_dir, "stderr.log"), "w") as f:
            f.write(e.stderr)
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"An unexpected error occurred during trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-config",
        type=str,
        default="hparam_search/configs/search_space.yaml",
        help="Path to the hyperparameter search configuration file.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="hparam_search/configs/base_hparams.yaml",
        help="Path to the base configuration file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=2,
        help="Number of trials for the hyperparameter search.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="sac-hparam-search-ConstantRewardEnv",
        help="Name for the Optuna study.",
    )

    args = parser.parse_args()

    # Create a directory for the study
    base_search_dir = os.path.join("hparam_search", "hparam_runs", args.study_name)
    counter = 1
    search_dir = f"{base_search_dir}_{counter}"
    # Each run must have a unique ID that is the counter appended to the base name
    while os.path.exists(search_dir):
        counter += 1
        search_dir = f"{base_search_dir}_{counter}"
    os.makedirs(search_dir, exist_ok=True)
    print(f"Hyperparameter search results will be saved to: {search_dir}")
    # Save the search config
    shutil.copy(args.search_config, os.path.join(search_dir, "search_config.yaml"))

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),  # Prune after 5 trials
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args, search_dir),
            n_trials=args.n_trials,
            timeout=600,
        )  # Added a timeout for safety
    except KeyboardInterrupt:
        print("Search interrupted by user.")

    print("\n\n--- Hyperparameter Search Finished ---")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    # Get the best trial
    try:
        best_trial = study.best_trial
        print("\n--- Best Trial ---")
        print(f"  Trial Number: {best_trial.number}")
        print(f"  Value (Final Avg Return): {best_trial.value:.4f}")

        print("\n  Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        best_trial_dir = os.path.join(search_dir, f"trial_{best_trial.number}")
        print(f"\nLogs and config for the best trial are in: {best_trial_dir}")

    except ValueError:
        print("No trials were completed. Could not determine the best trial.")

    # Save results to a file
    results_df = study.trials_dataframe()
    results_df.to_csv(f"{search_dir}/results.csv", index=False)
    print(f"\nFull results saved to {search_dir}/results.csv")

    # Move the optuna db to the search directory
    if os.path.exists(f"{args.study_name}.db"):
        shutil.move(f"{args.study_name}.db", search_dir)
