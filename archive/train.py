# train.py

import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from data.dataloaders.movie_dataset import get_movie_datasets, collate_fn # get_dataloaders_with_padding
from data.dataloaders.aeon_dataset import get_aeon_datasets
from models.base_model import BaseModel
from training.callbacks.metrics_logger import MetricsLogger  # Import the custom metrics logger
import json
from evaluation.metrics.metric_utils import compute_perturb_std

# ANSI color codes
RED = '\033[31m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'
BLACK_b = '\033[1;30m'	
RED_b = '\033[1;31m'	
GREEN_b = '\033[1;32m'	
YELLOW_b  = '\033[1;33m'	
BLUE_b  = '\033[1;34m'	
PURPLE_b  = '\033[1;35m'	
CYAN_b  = '\033[1;36m'	
WHITE_b = '\033[1;37m'
RESET_b = '\033[0;0m'

"""
-----------------------------
|        Example usage:     |
-----------------------------

from training.trainer import train_experiment

train_experiment(
    model_name='todynet',
    data_dir='path/to/your/data',
    experiment_name='todynet_experiment_1',
    seed=42,
    batch_size=16,
    hidden_dim=128,  # This might not be used directly; specify in todynet_params
    max_seq_len=180,  # Fixed sequence length
    optimizer='adam',
    lr=1e-4,
    weight_decay=1e-4,
    max_epochs=2000,
    gradient_clip_val=0.5,
    use_class_weights=True,
    use_lookahead=False,
    scheduler='reduce_on_plateau',
    scheduler_params={'mode': 'max', 'factor': 0.5, 'patience': 50},
    todynet_params={
        'num_layers': 3,
        'groups': 4,
        'pool_ratio': 0.2,
        'kern_size': [9, 5, 3],
        'in_dim': 64,
        'hidden_dim': 128,
        'out_dim': 256,
        'dropout': 0.5,
        'gnn_model_type': 'dyGIN2d',  # or 'dyGCN2d'
        'seq_length': 180,
        'num_nodes': input_dim  # Set this to your input_dim
    }
)
"""

def train_experiment(run, dataset_name, model_name, data_dir=None, experiment_name=None, aeon_dataset=None, **kwargs):
    # 1. Set seed for reproducibility
    #pl.seed_everything(kwargs.get('seed', 42))

    title = f"------Training Run Number {run}------"
    len_text = len(title)
    print(f"\n\n\n{PURPLE_b}{len_text*'-'}")
    print(f"------Training Run Number {run}------")
    print(f"{len_text*'-'}{RESET_b}\n")
    
    # 2. Input validation
    print("----------------------------")
    print("------Input Validation------")
    print("----------------------------\n")
    
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

    # Validate 'batch_size'
    batch_size = kwargs.get('batch_size', 32)
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

    # Validate 'max_seq_len'
    max_seq_len = kwargs.get('max_seq_len', 180)
    #if not isinstance(max_seq_len, int) or max_seq_len <= 0:
    #    print(f"{RED}[-]{RESET} 'max_seq_len' must be a positive integer.")
    #    raise ValueError("'max_seq_len' must be a positive integer.")
    #else:
    #    print(f"{GREEN}[+]{RESET} max_seq_len {BLUE}{max_seq_len}{RESET} is valid.")
    
    # 3. Get data loaders
    print("\n\n--------------------------")
    print("------Get DataLoader------")
    print("--------------------------\n")    

    # Get data loaders
    try:
        if data_dir:
            print(f"{YELLOW}[->]{RESET} loading data from {BLUE}{data_dir}{RESET}.")
            train_dataset, val_dataset, test_dataset, out_seq_len, out_num_classes, out_feat_in = get_movie_datasets(data_dir, max_seq_len=max_seq_len)
            train_feature_mapping, val_feature_mapping, test_feature_mapping = train_dataset.feature_mapping, val_dataset.feature_mapping, test_dataset.feature_mapping
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            # train_loader, val_loader, test_loader, out_seq_len, out_num_classes, out_feat_in = get_dataloaders_with_padding(data_dir, batch_size, max_seq_len)
            # num_classes = getattr(train_loader.dataset, 'num_classes', None)
            seq_len, num_classes, feats_size = out_seq_len, out_num_classes, out_feat_in
            if num_classes is None:
                print(f"{RED}[-]{RESET} The dataset does not have a 'num_classes' attribute.")
                raise AttributeError("The dataset does not have a 'num_classes' attribute.")
            else:
                print(f"{GREEN}[+]{RESET} Number of classes: {BLUE}{num_classes}{RESET}.")
        else:
            print(f"{YELLOW}[->]{RESET} loading data from {BLUE}{aeon_dataset}{RESET}.")
            train_dataset, val_dataset, test_dataset = get_aeon_datasets(aeon_dataset) # random_state=kwargs.get('seed', 42))
            seq_len,num_classes,feats_size =train_dataset.seq_len,train_dataset.num_classes,train_dataset.feat_in # change
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            num_classes = num_classes # getattr(train_dataset, 'num_classes', None) # change
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
    #print(f"{YELLOW}[->]{RESET} Feature Mapping:\n {BLUE}{json.dumps(train_feature_mapping, indent=4)}{RESET}.")
    #print(f"{YELLOW}[->]{RESET} Feature Mapping:\n {BLUE}{json.dumps(val_feature_mapping, indent=4)}{RESET}.")
    if data_dir:
        print(f"{YELLOW}[->]{RESET} Feature Mapping:\n {BLUE}{json.dumps(test_feature_mapping, indent=4)}{RESET}.")

    # 4. Get input dimension from dataset
    try:
        sample_input, _, _ = next(iter(train_loader))
        if sample_input.ndim < 2:
            print(f"{RED}[-]{RESET} Sample input does not have enough dimensions.")
            raise ValueError("Sample input does not have enough dimensions.")
        input_dim = sample_input.shape[-2]
        # print("input_dim is value:", input_dim)
        print(f"{GREEN}[+]{RESET} Input dimension {BLUE}{input_dim}{RESET} determined from dataset.")
    except StopIteration:
        print(f"{RED}[-]{RESET} The training data loader is empty.")
        raise ValueError("The training data loader is empty. Please check your data.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error determining input dimension: {e}")
        raise e

    # Update todynet_params with input_dim and seq_length
    if model_name == 'todynet':
        # Create a copy to avoid modifying the user's original dictionary
        user_todynet_params = kwargs.get('todynet_params', {})
        todynet_params = user_todynet_params.copy()

        # Remove keys that will be set explicitly to avoid duplicates
        for key in ['seq_length', 'num_nodes']: # 'in_dim', 
            todynet_params.pop(key, None)

        # Set the required parameters
        # todynet_params['in_dim'] = 64 # input_dim # change
        todynet_params['seq_length'] = seq_len # max_seq_len # change
        todynet_params['num_nodes'] = feats_size  # Set if required # input_dim # change
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

    # 6. Validate 'todynet_params' if 'model_name' is 'todynet'
    todynet_params = kwargs.get('todynet_params', {})
    if model_name == 'todynet':
        required_todynet_params = [
            'num_layers', 'groups', 'pool_ratio', 'kern_size', 'in_dim',
            'hidden_dim', 'out_dim', 'dropout', 'gnn_model_type', 'num_nodes', 'seq_length'
        ]
        missing_params = [p for p in required_todynet_params if p not in todynet_params]
        if missing_params:
            print(f"{RED}[-]{RESET} The following 'todynet_params' are missing: {BLUE}{missing_params}{RESET}")
            raise ValueError(f"The following 'todynet_params' are missing: {missing_params}")
        else:
            print(f"{GREEN}[+]{RESET} All required 'todynet_params' are provided.")
        # Validate 'kern_size' is a list of integers
        kern_size = todynet_params.get('kern_size')
        if not isinstance(kern_size, list) or not all(isinstance(k, int) for k in kern_size):
            print(f"{RED}[-]{RESET} 'kern_size' in 'todynet_params' must be a list of integers.")
            raise ValueError("'kern_size' in 'todynet_params' must be a list of integers.")
        else:
            print(f"{GREEN}[+]{RESET} 'kern_size' is valid.")

    # Experiment Repository Setup
    print("\n\n---------------------------------")
    print("------Experiment Repository------")
    print("---------------------------------\n")

    # 7. Ensure directories exist
    # Create experiment directory
    experiment_dir = os.path.join(f'experiments/experiments/experiments_{dataset_name}', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"{GREEN}[+]{RESET} created experiment-dir @\t'{BLUE}{experiment_dir}{RESET}'.")    
    
    # Create subdirectories
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"{GREEN}[+]{RESET} created log-dir @\t'{BLUE}{log_dir}{RESET}'.")

    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"{GREEN}[+]{RESET} created checkpoint-dir @\t'{BLUE}{checkpoint_dir}{RESET}'.")

    # Initialize Model
    print("\n\n----------------------------")
    print("------Initialize Model------")
    print("----------------------------\n")

    if model_name == 'todynet':                           
        print(f"Input dim is: {todynet_params['in_dim']}")
    print(f"{GREEN}[+]{RESET} Initialized model with num_classes: {BLUE_b}{num_classes}{RESET_b}")
    print(f"{GREEN}[+]{RESET} Initialized model with feats_size: {BLUE_b}{feats_size}{RESET_b}")
    print(f"{GREEN}[+]{RESET} Initialized model with seq_len: {BLUE_b}{seq_len}{RESET_b}")


    # 8. Initialize model
    try:
        model = BaseModel(
            model_name=model_name,
            num_classes=num_classes, # 
            input_dim= feats_size,# kwargs.get('input_dim', 64), # input_dim # change
            hidden_dim=kwargs.get('hidden_dim', 128),
            max_seq_len=seq_len, # max_seq_len # change
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=kwargs.get('bidirectional', False),
            optimizer_name=kwargs.get('optimizer', 'adamw'),
            lr=kwargs.get('lr', 1e-3),
            weight_decay=kwargs.get('weight_decay', 1e-5),
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

        # # Move model to device
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)

        # # Compute FLOPS
        # print(f"\n{GREEN}[+]{RESET} Computing FLOPS and parameter count...")
        # model.compute_flops()

    except Exception as e:
        print(f"{RED}[-]{RESET} Error initializing model: {e}")
        raise e

    # 9. Set up callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=kwargs.get('early_stop_patience', 50),
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Include metrics logger
    metrics_logger = MetricsLogger(experiment_dir=experiment_dir)

    print(f"{GREEN}[+]{RESET} Callbacks set up successfully.")

    # Initialize Training
    print("\n\n-------------------------------")
    print("------Initialize Training------")
    print("-------------------------------\n")

    print("check train loader shape")
    # Get a sample input
    sample_input, _, _ = next(iter(train_loader))
    print(f"Sample input shape: {sample_input.shape}")

    # 10. Initialize trainer
    try:
        trainer = pl.Trainer(
            max_epochs=kwargs.get('max_epochs', 50),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=kwargs.get('devices', 1 if torch.cuda.is_available() else None),
            callbacks=[early_stop_callback, checkpoint_callback, metrics_logger],
            gradient_clip_val=kwargs.get('gradient_clip_val', 0.5),
            default_root_dir=experiment_dir,
            logger=pl.loggers.TensorBoardLogger(save_dir=log_dir, name=''),
        )
        print(f"{GREEN}[+]{RESET} Trainer initialized successfully.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error initializing trainer: {e}")
        raise e

    # 11. Train the model
    print(f"\n{GREEN}[+]{RESET} Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        print(f"{GREEN}[+]{RESET} Training completed successfully.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error during training: {e}")
        raise RuntimeError(f"Error during training: {e}")

    # Initialize Testing
    print("\n\n------------------------------")
    print("------Initialize Testing------")
    print("------------------------------\n")

    # 12. Test the model
    print(f"\n{GREEN}[+]{RESET} Starting testing...")
    try:
        test_results = trainer.test(model, test_loader)
        test_results_df = pd.DataFrame(test_results)
        result_dir = os.path.join(experiment_dir, f"run_{run}_test_metric_values.csv")
        test_results_df.to_csv(result_dir, index=False)
        print(f"{YELLOW}[->]{RESET} Test results saved to {BLUE_b}{result_dir}{RESET_b}")
        print(f"{GREEN}[+]{RESET} Testing completed successfully.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error during testing: {e}")
        raise RuntimeError(f"Error during testing: {e}")


    # Initialize Interpretability Analysis
    print("\n\n------------------------------------------------")
    print("------Initialize Interpretability Analysis------")
    print("------------------------------------------------\n")

    # 13. After testing is complete
    print(f"\n{GREEN}[+]{RESET} Starting interpretability analysis...")
    try:
        # Dynamically compute perturb_std based on the training or test dataset
        perturb_std = compute_perturb_std(train_dataset)
        print(f"{GREEN}[+]{RESET} Computed perturb_std for test dataset: {GREEN_b}{perturb_std}{RESET_b}")
        
        model.interpret_model_using_gradients(test_loader, num_samples=1, save_visualizations=True, test_feature_mapping=test_feature_mapping, perturb_std=perturb_std)
        print(f"{GREEN}[+]{RESET} Interpretability analysis IntegratedGradients successfully finished.")
        model.interpret_model_using_deeplift(test_loader, num_samples=1, save_visualizations=True, test_feature_mapping=test_feature_mapping, perturb_std=perturb_std)
        print(f"{GREEN}[+]{RESET} Interpretability analysis DeepLift successfully finished.")
        
        print(f"{GREEN}[+]{RESET} Interpretability analysis completed successfully.")
    except Exception as e:
        print(f"{RED}[-]{RESET} Error during interpretability analysis: {e}")
        raise RuntimeError(f"Error during interpretability analysis: {e}")