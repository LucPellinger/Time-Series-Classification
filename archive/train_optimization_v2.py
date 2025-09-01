# train.py

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from data.dataloaders.movie_dataset import get_movie_datasets, collate_fn  # get_dataloaders_with_padding
from data.dataloaders.aeon_dataset import get_aeon_datasets
from models.base_model import BaseModel
from training.callbacks.metrics_logger import MetricsLogger  # Import the custom metrics logger
import json

# Import Optuna
import optuna
from optuna.samplers import TPESampler
# from optuna.integration import PyTorchLightningPruningCallback
from optuna.storages import RDBStorage

# ANSI color codes
RED = '\033[31m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'
BLACK_b = '\033[1;30m'
RED_b = '\033[1;31m'
GREEN_b = '\033[1;32m'
YELLOW_b = '\033[1;33m'
BLUE_b = '\033[1;34m'
PURPLE_b = '\033[1;35m'
CYAN_b = '\033[1;36m'
WHITE_b = '\033[1;37m'
RESET_b = '\033[0;0m'

def train_experiment_optimization(model_name, data_dir=None, experiment_name=None, aeon_dataset=None, max_trials=50, db_url='sqlite:////content/drive/MyDrive/Colab Notebooks/work_project/modelling_new/optimization_experiments/new_optuna_study.db',**kwargs):
    # Import necessary modules
    import optuna
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import shutil

    # Initialize todynet_params to None
    todynet_params = None

    # 1. Set seed for reproducibility
    #pl.seed_everything(kwargs.get('seed', 42))

    # 2. Input validation
    print("\n\n-------------------------------------------------------------")
    print("------Input Validation before Optimization using Optuna------")
    print("-------------------------------------------------------------\n")

    # Check that 'experiment_name' is provided and is a string
    if not experiment_name or not isinstance(experiment_name, str):
        print(f"{RED}[-]{RESET} 'experiment_name' must be provided and must be a non-empty string.")
        raise ValueError("'experiment_name' must be provided and must be a non-empty string.")
    else:
        print(f"{GREEN}[+]{RESET} experiment_name {BLUE}{experiment_name}{RESET} provided in correct format.")

    # Ensure 'model_name' is supported
    supported_models = ['lstm', 'lstm_classifier', 'timemil', 'todynet', 'timemil_extended']
    if model_name not in supported_models:
        print(f"{RED}[-]{RESET} 'model_name' must be one of {supported_models}.")
        raise ValueError(f"'model_name' must be one of {supported_models}.")
    else:
        print(f"{GREEN}[+]{RESET} model_name {BLUE}{model_name}{RESET} is supported.")

    # Ensure that either 'data_dir' or 'aeon_dataset' is provided
    if data_dir is None and aeon_dataset is None:
        print(f"{RED}[-]{RESET} Either 'data_dir' or 'aeon_dataset' must be provided.")
        raise ValueError("Either 'data_dir' or 'aeon_dataset' must be provided.")
    else:
        data_source = data_dir if data_dir else aeon_dataset
        print(f"{GREEN}[+]{RESET} Data source {BLUE}{data_source}{RESET} provided.")

    # Check if 'data_dir' exists
    if data_dir:
        if os.path.exists(data_dir):
            print(f"{GREEN}[+]{RESET} Path {BLUE}{data_dir}{RESET} exists.")
            print("\tContinue with batch_size check.")
        else:
            print(f"{RED}[-]{RESET} Path {BLUE}{data_dir}{RESET} does not exist.")
            raise FileNotFoundError(f"Path {data_dir} does not exist.")

    # Set up the RDBStorage
    print("\n\n---------------------------------")
    print("------Setting up RDBStorage------")
    print("---------------------------------\n")

    try:
        storage = RDBStorage(url=db_url)
        print(f"{GREEN}[+]{RESET} RDBStorage initialized successfully with database URL: {BLUE}{db_url}{RESET}.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error initializing RDBStorage: {e}")
        raise e

    # Set up the Optuna study
    study_name = f"{experiment_name}_optimization"  # Unique study name for this experiment
    sampler = TPESampler(multivariate=True, 
                        **TPESampler.hyperopt_parameters())

    try:
        study = optuna.create_study(
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            direction='minimize',  # Change 'minimize' to 'maximize' if optimizing for accuracy
            load_if_exists=True  # Load the study if it already exists
        )
        print(f"{GREEN}[+]{RESET} Optuna study '{BLUE}{study_name}{RESET}' created successfully.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error creating Optuna study: {e}")
        raise e

    # Define the objective function
    def objective(trial):
        
        title = f"------Trial number Validation {trial.number}------"
        len_text = len(title)
        print(f"\n\n\n{PURPLE_b}{len_text*'-'}")
        print(f"------Trial number Validation {trial.number}------")
        print(f"{len_text*'-'}{RESET_b}\n")

        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        max_seq_len = trial.suggest_int('max_seq_len', 28, 100)
        kwargs.update({'max_seq_len': max_seq_len})
        print(f"{GREEN}[+]{RESET}Selected batch_size {batch_size} & max_seq_len {max_seq_len}")

        # 3. Get data loaders
        print("\n\n--------------------------")
        print("------Get DataLoader------")
        print("--------------------------\n")

        # Get data loaders
        try:
            if data_dir:
                print(f"{YELLOW}[->]{RESET} loading data from {BLUE}{data_dir}{RESET}.")
                train_dataset, val_dataset, test_dataset, out_seq_len, out_num_classes, out_feat_in = get_movie_datasets(data_dir, max_seq_len=kwargs.get('max_seq_len', 180))
                train_feature_mapping, val_feature_mapping, test_feature_mapping = train_dataset.feature_mapping, val_dataset.feature_mapping, test_dataset.feature_mapping
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                seq_len, num_classes, feats_size = out_seq_len, out_num_classes, out_feat_in
                if num_classes is None:
                    print(f"{RED}[-]{RESET} The dataset does not have a 'num_classes' attribute.")
                    raise AttributeError("The dataset does not have a 'num_classes' attribute.")
                else:
                    print(f"{GREEN}[+]{RESET} Number of classes: {BLUE}{num_classes}{RESET}.")
            else:
                print(f"{YELLOW}[->]{RESET} loading data from {BLUE}{aeon_dataset}{RESET}.")
                train_dataset, val_dataset, test_dataset = get_aeon_datasets(aeon_dataset) # , random_state=kwargs.get('seed', 42))
                seq_len, num_classes, feats_size = train_dataset.seq_len, train_dataset.num_classes, train_dataset.feat_in
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                if num_classes is None:
                    print(f"{RED}[-]{RESET} The Aeon dataset does not have a 'num_classes' attribute.")
                    raise AttributeError("The Aeon dataset does not have a 'num_classes' attribute.")
                else:
                    print(f"{GREEN}[+]{RESET} Number of classes: {BLUE}{num_classes}{RESET}.")
        except Exception as e:
            print(f"{RED}[-]{RESET} Error loading data: {e}")
            raise e

        print(f"{YELLOW}[->]{RESET} seq_len is {BLUE}{seq_len}{RESET}.")
        print(f"{YELLOW}[->]{RESET} num_classes is {BLUE}{num_classes}{RESET}.")
        print(f"{YELLOW}[->]{RESET} feats_size is {BLUE}{feats_size}{RESET}.")
        print(train_feature_mapping)
        print("\n\n")      

        if model_name == "timemil" or model_name == "timemil_extended":

          todynet_params = None

          # Suggest hyperparameters
          #batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
          # hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
          lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
          weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
          dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
          num_layers = trial.suggest_int('num_layers', 1, 5)
          #max_seq_len = trial.suggest_int('max_seq_len', 28, 100)
          # Any other hyperparameters you want to optimize
          gradient_clip_val = trial.suggest_loguniform('gradient_clip_val', 0.1, 1.0)

          hidden_dim = 128  # Fixed for now

        elif model_name == "todynet":
          batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
          hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
          lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
          weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
          dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
          num_layers = trial.suggest_int('num_layers', 1, 5)
          #max_seq_len = trial.suggest_int('max_seq_len', 28, 100)
          gradient_clip_val = trial.suggest_loguniform('gradient_clip_val', 0.1, 1.0)
          pool_ratio = trial.suggest_uniform('pool_ratio', 0.1, 0.5)
          hidden_dim_tody = trial.suggest_int('hidden_dim_tody', 64, 256, step=64)
          output_dim_tody = trial.suggest_int('output_dim_tody', 64, 256, step=64)
          dropout_tody = trial.suggest_uniform('dropout_tody', 0.1, 0.5)
          
          # Define kern_size as a list of integers
          kern_size = trial.suggest_categorical('kern_size', [
              [9, 5, 3], 
              [7, 5, 3], 
              [11, 7, 5]
          ])
  
          # Update todynet_params
          todynet_params = kwargs.get('todynet_params', {}).copy()
          todynet_params.update({
              'num_layers': 3,
              'groups': 1,
              'gnn_model_type': 'dyGCN2d',
              'in_dim': 1,
              'hidden_dim': hidden_dim_tody,
              'out_dim': output_dim_tody,
              'dropout': dropout_tody,
              'pool_ratio': pool_ratio,
              'kern_size': kern_size,  # Include the suggested kern_size
              'seq_length': max_seq_len,  # Derived from dataset
              'num_nodes': feats_size  # Derived from dataset
          })
  
          kwargs['todynet_params'] = todynet_params
          

        elif model_name == "lstm_classifier":
          # Suggest hyperparameters
          #batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
          hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
          lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
          weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
          dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
          num_layers = trial.suggest_int('num_layers', 1, 5)
          #max_seq_len = trial.suggest_int('max_seq_len', 28, 100)
          gradient_clip_val = trial.suggest_loguniform('gradient_clip_val', 0.1, 1.0)
          # Any other hyperparameters you want to optimize
          todynet_params = None

        # Update kwargs with suggested hyperparameters
        kwargs.update({
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'num_layers': num_layers,
            # 'max_seq_len': max_seq_len,
            'gradient_clip_val': gradient_clip_val,
        })


        # Validate 'batch_size'
        if not isinstance(batch_size, int) or batch_size <= 0:
            print(f"{RED}[-]{RESET} 'batch_size' must be a positive integer.")
            raise ValueError("'batch_size' must be a positive integer.")
        else:
            print(f"{GREEN}[+]{RESET} batch_size {BLUE}{batch_size}{RESET} is valid.")

        # Validate 'use_class_weights'
        use_class_weights = kwargs.get('use_class_weights', False)
        if not isinstance(use_class_weights, bool):
            print(f"{RED}[-]{RESET} 'use_class_weights' must be a boolean.")
            raise ValueError("'use_class_weights' must be a boolean.")
        else:
            print(f"{GREEN}[+]{RESET} use_class_weights {BLUE}{use_class_weights}{RESET} is valid.")


        # 4. Update todynet_params with input_dim and seq_length
        if model_name == 'todynet':
            # Create a copy to avoid modifying the user's original dictionary
            user_todynet_params = kwargs.get('todynet_params', {})
            todynet_params = user_todynet_params.copy()

            # Remove keys that will be set explicitly to avoid duplicates
            for key in ['seq_length', 'num_nodes']:  # 'in_dim',
                todynet_params.pop(key, None)

            # Set the required parameters
            todynet_params['seq_length'] = seq_len
            todynet_params['num_nodes'] = feats_size
            # Update kwargs
            kwargs['todynet_params'] = todynet_params

        # 5. Get class weights if 'use_class_weights' is True
        class_weights = None
        if use_class_weights:
            if hasattr(train_loader.dataset, 'class_weights'):
                class_weights = torch.tensor(train_loader.dataset.class_weights, dtype=torch.float)
                print(f"{GREEN}[+]{RESET} Class weights obtained from dataset.")
            else:
                print(f"{RED}[-]{RESET} Dataset does not have 'class_weights' attribute, but 'use_class_weights' is True.")
                raise AttributeError("Dataset does not have 'class_weights' attribute, but 'use_class_weights' is True.")

        # Experiment Repository Setup
        print("\n\n---------------------------------")
        print("------Experiment Repository------")
        print("---------------------------------\n")

        # 6. Ensure directories exist
        # Create experiment directory for the trial
        trial_experiment_name = f"{experiment_name}_trial_{trial.number}"
        experiment_dir = os.path.join('experiments/optimization_experimennts', trial_experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"{GREEN}[+]{RESET} created experiment-dir @\t'{BLUE}{experiment_dir}{RESET}'.")

        # Create subdirectories
        log_dir = os.path.join(experiment_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        print(f"{GREEN}[+]{RESET} created log-dir @\t'{BLUE}{log_dir}{RESET}'.")

        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"{GREEN}[+]{RESET} created checkpoint-dir @\t'{BLUE}{checkpoint_dir}{RESET}'.")

        # 7. Initialize Model
        print("\n\n----------------------------")
        print("------Initialize Model------")
        print("----------------------------\n")

        if model_name == 'todynet':
            print(f"Input dim is: {todynet_params['in_dim']}")
        print(f"{GREEN}[+]{RESET} Initialized model with num_classes: {BLUE_b}{num_classes}{RESET_b}")
        print(f"{GREEN}[+]{RESET} Initialized model with feats_size: {BLUE_b}{feats_size}{RESET_b}")
        print(f"{GREEN}[+]{RESET} Initialized model with seq_len: {BLUE_b}{seq_len}{RESET_b}")

        # Initialize model
        try:
            model = BaseModel(
                model_name=model_name,
                num_classes=num_classes,
                input_dim=feats_size,
                hidden_dim=hidden_dim,
                max_seq_len=seq_len,
                num_layers=num_layers,
                dropout=dropout,
                optimizer_name=kwargs.get('optimizer', 'adamw'),
                lr=lr,
                weight_decay=weight_decay,
                scheduler_name=kwargs.get('scheduler', None),
                scheduler_params=kwargs.get('scheduler_params', {}),
                use_lookahead=kwargs.get('use_lookahead', False),
                lookahead_params=kwargs.get('lookahead_params', {}),
                use_class_weights=use_class_weights,
                class_weights=class_weights,
                todynet_params=todynet_params,
                experiment_name=experiment_name,  # Pass experiment_name to the model
                experiment_dir=experiment_dir
            )
            print(f"{GREEN}[+]{RESET} Model initialized successfully.")
        except Exception as e:
            print(f"{RED}[-]{RESET} Error initializing model: {e}")
            raise e

        # 8. Set up callbacks
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=kwargs.get('early_stop_patience', 20),
            verbose=True
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        #pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        # Include metrics logger
        metrics_logger = MetricsLogger(experiment_dir=experiment_dir)

        print(f"{GREEN}[+]{RESET} Callbacks set up successfully.")

        # 9. Initialize trainer
        print("\n\n-------------------------------")
        print("------Initialize Training------")
        print("-------------------------------\n")

        # Initialize trainer
        try:
            trainer = pl.Trainer(
                max_epochs=kwargs.get('max_epochs', 50),
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=kwargs.get('devices', 1 if torch.cuda.is_available() else None),
                callbacks=[early_stop_callback, checkpoint_callback, metrics_logger], # pruning_callback
                gradient_clip_val=kwargs.get('gradient_clip_val', 0.5),
                default_root_dir=experiment_dir,
                logger=pl.loggers.TensorBoardLogger(save_dir=log_dir, name=''),
            )
            print(f"{GREEN}[+]{RESET} Trainer initialized successfully.")
        except Exception as e:
            print(f"{RED}[-]{RESET} Error initializing trainer: {e}")
            raise e

        # 10. Train the model
        print(f"\n{GREEN}[+]{RESET} Starting training...")
        try:
            trainer.fit(model, train_loader, val_loader)
            print(f"{GREEN}[+]{RESET} Training completed successfully.")
        except Exception as e:
            print(f"{RED}[-]{RESET} Error during training: {e}")
            raise RuntimeError(f"Error during training: {e}")

       

        print(f"{GREEN}[+]{RESET} Training completed successfully.")

        # Get validation loss
        val_loss = trainer.callback_metrics['val_loss'].item()
        print(f"{GREEN}[+]{RESET} Validation Loss: {BLUE}{val_loss}{RESET}")
        if val_loss is None:
            raise optuna.TrialPruned()

        # Cleanup to save space
        shutil.rmtree(experiment_dir)

        return val_loss

    # Define a callback function to display trial results
    def print_trial_result(study, trial):
        print(f"{GREEN_b}Trial {trial.number}{RESET_b} finished with value: {trial.value} and parameters: {trial.params}")


    # Run the optimization
    study.optimize(
        objective,
        n_trials=max_trials,
        callbacks=[print_trial_result],
        show_progress_bar=True
    )

    # After optimization, you can get the best hyperparameters
    print(f'\n\n\n{GREEN_b}{30*"-"}{RESET_b}')
    print(f'{GREEN_b}[+]{RESET_b}Best hyperparameters:', study.best_params)
    print(f'{GREEN_b}{30*"-"}{RESET_b}')

    # Optionally, you can retrain the model with the best hyperparameters
    best_params = study.best_params
    kwargs.update(best_params)
    # Set batch_size back to the best one
    #kwargs['batch_size'] = best_params['batch_size']
    #train_experiment(model_name, data_dir=data_dir, experiment_name=experiment_name + '_best', aeon_dataset=aeon_dataset, **kwargs)

