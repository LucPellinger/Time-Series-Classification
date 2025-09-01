    # models.py
import os
import pandas as pd
import yaml
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# model imports
from architectures.mil_model.timemil import TimeMIL  # Adjust the import path as needed
from architectures.lstm_model.lstm_model  import LSTMClassifier  # Import the LSTM model
from architectures.todynet_model.tody_net import TodyNetClassifier # import the TodyNetClassifier
from architectures.mil_model_extended.timemil import TimeMIL_extended

# import utility methods for visualization
from evaluation.interpretability.visualizations import visualize_attributions, visualize_attention_weights
from evaluation.metrics.metric_utils import compute_global_feature_importance, \
                  visualize_feature_importance, compute_entropy, compute_sparsity
                  #compute_feature_interaction, \
                  #visualize_feature_interaction

# interpretability
from captum.attr import IntegratedGradients, Saliency, DeepLift, ShapleyValues
from captum.metrics import infidelity, sensitivity_max
import captum
# import shap

import matplotlib.pyplot as plt
import numpy as np

# ANSI color codes
BLACK = '\033[0;30m'	
RED = '\033[0;31m'	
GREEN = '\033[0;32m'	
YELLOW  = '\033[0;33m'	
BLUE  = '\033[0;34m'	
PURPLE  = '\033[0;35m'	
CYAN  = '\033[0;36m'	
WHITE = '\033[0;37m'
RESET = '\033[0;0m'

BLACK_b = '\033[1;30m'	
RED_b = '\033[1;31m'	
GREEN_b = '\033[1;32m'	
YELLOW_b  = '\033[1;33m'	
BLUE_b  = '\033[1;34m'	
PURPLE_b  = '\033[1;35m'	
CYAN_b  = '\033[1;36m'	
WHITE_b = '\033[1;37m'
RESET_b = '\033[0;0m'

BLACK_ = '\e[4;30m'	
RED_ = '\e[4;31m'	
GREEN_ = '\e[4;32m'	
YELLOW_  = '\e[4;33m'	
BLUE_  = '\e[4;34m'	
PURPLE_  = '\e[4;35m'	
CYAN_  = '\e[4;36m'	
WHITE_ = '\e[4;37m'
RESET_ = '\e[0;0m'

# class SHAPModelWrapper(nn.Module):
#     def __init__(self, model, model_name, lengths=None, target_class=None):
#         """
#         Wrapper to make the model compatible with SHAP's DeepExplainer.
        
#         Args:
#             model (nn.Module): The underlying PyTorch model.
#             model_name (str): Name of the model (timemil, todynet, etc.).
#             lengths (torch.Tensor): Additional argument for models that require sequence lengths.
#         """
#         super().__init__()
#         self.model = model
#         self.model_name = model_name
#         self.lengths = lengths
#         self.target_class = target_class

#     def forward(self, x):
#         """
#         Forward pass to return scalar probabilities for the target class.

#         Args:
#             x (torch.Tensor): Input tensor.
#         Returns:
#             torch.Tensor: Scalar probabilities for the target class.
#         """
#         if self.model_name == 'timemil':
#             _, probs = self.model(x)
#         elif self.model_name == 'lstm_classifier':
#             _, probs = self.model(x, self.lengths)
#         elif self.model_name == 'todynet':
#             _, probs, _, _ = self.model(x)
#         else:
#             raise ValueError(f"Model {self.model_name} is not supported for SHAP.")

#         # Select the probability for the target class (scalar output)
#         if self.target_class is not None:
#             return probs[:, self.target_class]  # Scalar probability for target class
#         return probs



class BaseModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, input_dim, 
                  hidden_dim=128, max_seq_len=365, num_layers=2, 
                  dropout=0.2, bidirectional=False,
                  optimizer_name='adamw', lr=1e-3, weight_decay=1e-5,
                  scheduler_name=None, scheduler_params=None,
                  use_lookahead=False, lookahead_params=None,
                  use_class_weights=False, class_weights=None,
                  todynet_params=None,
                  experiment_name=None,
                  experiment_dir=None):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for reproducibility

        # Save the experiment name and directory
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir

        # Save hyperparameters to a YAML file
        hparams_file = os.path.join(self.experiment_dir, 'hparams.yaml')
        os.makedirs(os.path.dirname(hparams_file), exist_ok=True)
        with open(hparams_file, 'w') as f:
            yaml.dump(self.hparams, f)

        # Initialize lists to store predictions and labels
        self.test_preds = []
        self.test_probs = []
        self.test_labels = []

        self.model_name = model_name
        
        # Model-specific parameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # LSTM-specific parameters
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers=num_layers
         
        # training hyperparameters
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params or {}
        self.use_lookahead = use_lookahead
        self.lookahead_params = lookahead_params or {}
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        
        # TodyNet parameters
        self.todynet_params = todynet_params or {}        
        
        # Define loss criterion
        if self.use_class_weights and self.class_weights is not None:
            print(f"{GREEN}[+]{RESET}Using weighted CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average='macro')

        # Define model architecture
        self.build_model()
        {RESET_b}
        print(f"{PURPLE_b}\n\nScheduler name:{RESET_b}\t\t", self.scheduler_name)
        print(f"{PURPLE_b}Scheduler parameters:{RESET_b}\t", self.scheduler_params)
        print(f"{PURPLE_b}Use Lookahead:{RESET_b}\t\t", self.use_lookahead)
        print(f"{PURPLE_b}Lookahead parameters:{RESET_b}\t", self.lookahead_params)
        print(f"{PURPLE_b}Use Class Weights:{RESET_b}\t", self.use_class_weights)
        print(f"{PURPLE_b}Class Weights:{RESET_b}\t\t", self.class_weights)
        print(f"{PURPLE_b}TodyNet parameters:{RESET_b}\t", self.todynet_params)
        print(f"{PURPLE_b}Model Name:{RESET_b}\t\t", self.model_name)
        print(f"{PURPLE_b}experiment_name:{RESET_b}\t", self.experiment_name)
        print(f"{PURPLE_b}experiment_dir:{RESET_b}\t\t", self.experiment_dir)
        print(f"{PURPLE_b}hparams_file:{RESET_b}\t\t", hparams_file)
        print(f"{PURPLE_b}input_dim:{RESET_b}\t\t", self.input_dim)
        print(f"{PURPLE_b}num_classes:{RESET_b}\t\t", self.num_classes)
        print(f"{PURPLE_b}max_seq_len:{RESET_b}\t\t", self.max_seq_len)
        print(f"{PURPLE_b}dropout:{RESET_b}\t\t", self.dropout)
        print(f"{PURPLE_b}hidden_dim:{RESET_b}\t\t", self.hidden_dim)
        print(f"{PURPLE_b}bidirectional:{RESET_b}\t\t", self.bidirectional)
        print(f"{PURPLE_b}num_layers:{RESET_b}\t\t", self.num_layers)
        print(f"{PURPLE_b}optimizer_name:{RESET_b}\t\t", self.optimizer_name)
        print(f"{PURPLE_b}lr:{RESET_b}\t\t\t", self.lr)
        print(f"{PURPLE_b}weight_decay:{RESET_b}\t\t", self.weight_decay)
        print(f"{PURPLE_b}criterion:{RESET_b}\t\t", self.criterion)
        print(f"{PURPLE_b}class_weights:{RESET_b}\t\t", self.class_weights)
        print(f"{RESET_b}\n\n")
    
    def interpret_model_using_gradients(self, data_loader, num_samples=10, save_visualizations=True, test_feature_mapping=None, perturb_std=None):
          self.eval()
          attr_list = []
          save_path = os.path.join(self.experiment_dir, "metrics") if save_visualizations else None
          os.makedirs(save_path, exist_ok=True)
          infidelity_scores = []
          #sensitivity_scores = []

          if self.model_name == 'timemil':
              attention_weights = self.attention_weights.cpu().numpy()
              print("shape attention weights output of model: ", attention_weights.shape)
          # elif self.model_name == 'todynet':
          #     print("Length Intermediate outputs: ", len(self.intermediate_outputs))
          #     print("DataType Intermediate outputs: ", type(self.intermediate_outputs))
          #     print("Example Intermediate outputs: ", type(self.intermediate_outputs[0]))
          #     print("Example Intermediate outputs: ", self.intermediate_outputs[0])
          #     intermediate_outputs = self.intermediate_outputs.cpu().numpy()
          #     print("shape intermediate_outputs output of model: ", intermediate_outputs.shape)
          
          for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            x, y, lengths = batch
            x = x.to(self.device)
            print("Shape of x in batch: ", x.shape)
            y = y.to(self.device)
            print("Shape of y in batch: ", y.shape)
            lengths = lengths.to(self.device) if lengths is not None else None
            print("Shape of lengths in batch: ", lengths.shape)
            # target = y[0].item()  # Get the first sample's target
            
            # Decode one-hot encoded labels into scalar indices
            y_indices = torch.argmax(y, dim=1)  # Convert one-hot to class indices
            target = y_indices[0].item()  # Get the first sample's target as a scalar

            def perturb_fn(inputs):
                perturb_std_expanded = perturb_std.view(1, 1, -1).to(inputs.device)
                noise = torch.randn_like(inputs) * perturb_std_expanded
                perturbed_inputs = inputs + noise
                return noise, perturbed_inputs  # Swap the order here

            # Define a wrapper function based on the model
            if self.model_name == 'timemil' or self.model_name == 'timemil_extended':
                def model_forward(input_x, *args):
                    logits, _ = self.model(input_x)
                    return logits
                additional_args = None
            elif self.model_name == 'lstm_classifier':
                def model_forward(input_x, lengths, *args):
                    logits, _ = self.model(input_x, lengths)
                    return logits
                additional_args = lengths
            elif self.model_name == 'todynet':
                def model_forward(input_x, *args):
                    input_x = input_x.permute(0, 2, 1)  # [batch_size, num_nodes, seq_length]
                    input_x = input_x.unsqueeze(1) # [batch_size, 1, num_nodes, seq_length]
                    logits, _, _ = self.model(input_x)
                    return logits
                additional_args = None
            else:
                raise ValueError(f"Model {self.model_name} not supported for interpretability.")

            ig = IntegratedGradients(model_forward)

            print(f"Input shape before IG: {x.shape}")
            print("Calculating integrated gradients using Captum")
            attr, delta = ig.attribute(
                inputs=x,
                target=target,
                additional_forward_args=additional_args,
                return_convergence_delta=True,
                n_steps=50,
                internal_batch_size=x.shape[0]  # Set to batch size
            )
            
            print("Target values", target)
            
            # Compute Infidelity
            infidelity_score = infidelity(model_forward, 
                                          perturb_fn, 
                                          x, 
                                          attr, 
                                          target=target,
                                          additional_forward_args=(additional_args,), 
                                          normalize=True
            )
            print(f"\n\nInfidelity score tensor shape: {infidelity_score.shape}, value: {infidelity_score}")
            infidelity_scores.extend(infidelity_score.cpu().tolist())  # Convert to list for easy handling
            
            # Compute Sensitivity
            # sensitivity_score = sensitivity_max(
            #     explanation_func=ig.attribute,
            #     #explanation_func=lambda *args, **kwargs: ig.attribute(
            #     #  *args, 
            #     #  **kwargs, 
            #     #  internal_batch_size=1, 
            #     #  additional_forward_args=additional_args
            #     #),  # Reduce internal batch size
            #     inputs=x,  # Input tensor
            #     target=target,  # Target classes for the inputs
            #     perturb_func=perturb_fn,
            #     n_steps=10,  # Number of steps for Integrated Gradients
            #     perturb_radius=0.02,  # Epsilon for perturbations
            #     n_perturb_samples=5,  # Number of perturbations per input
            #     norm_ord='fro',  # Frobenius norm
            #     max_examples_per_batch=1,  # Process one example per batch
            # )
            #print(f"\n\nInfidelity score tensor shape: {sensitivity_score.shape}, value: {sensitivity_score}")
            #sensitivity_scores.append(sensitivity_score.item())

            inputs = x.cpu().detach().numpy()
            targets = y_indices.cpu().detach().numpy()            
            attr = attr.cpu().detach().numpy()
            
            # attr_list.append((x.cpu().numpy(), attr, target))
            attr_list.append((x.cpu().numpy(), attr, target))
            print("Finished calculating integrated gradients using Captum.")
            print("attr shape:", attr.shape)
            print("inputs shape:", x.cpu().numpy().shape)
            #print("target shape:", target.shape)

            if i == 0:
                channels_last_in = True

                if x.shape[0] == self.input_dim:
                    channels_last_in=False
                elif x.shape[1] == self.input_dim:
                    channels_last_in=True
                else:
                    print(f"Warning: Unable to determine channels_last_in. Defaulting to True.")

                print("Value of channels_last_in:", channels_last_in)
                print("Creating visualization of IntegratedGradients as Saliency Map using Captum.")
                print("test_feature_mapping:", test_feature_mapping)
                # Visualize attributions
                if save_visualizations and i < 1:
                    visualize_attributions(attr, 
                                          inputs, 
                                          targets, 
                                          save_path=save_path, 
                                          num_samples=1, 
                                          channels_last=channels_last_in, 
                                          method_used="IntegratedGradients",
                                          test_feature_mapping=test_feature_mapping)
          
          # Compute global feature importance using Integrated Gradients attributions
          ig_global_importance = compute_global_feature_importance(attr_list, method="mean")
          np.save(os.path.join(self.experiment_dir, f"{self.model_name}_IG_Global_Importance.npy"), ig_global_importance)
          print(f"Saved global feature importance for Integrated Gradients at '{self.experiment_dir}'")

          visualize_feature_importance(ig_global_importance, method="IntegratedGradient", aggregation="mean", test_feature_mapping=test_feature_mapping, save_path=save_path)

          # Example usage for Integrated Gradients
          ig_entropies = compute_entropy(attr_list)
          avg_entropy_ig = np.mean(ig_entropies)
          print(f"{GREEN}[+]{RESET}Average Entropy (IntegratedGradients): {BLUE_b}{avg_entropy_ig}{RESET_b}")
          
          # Example: Compute sparsity for Integrated Gradients
          ig_sparsities = compute_sparsity(attr_list, method="IntegratedGradients", threshold=0.5)
          avg_sparsity_ig = np.mean(ig_sparsities)
          print(f"{GREEN}[+]{RESET}Average Sparsity (Integrated Gradients): {BLUE_b}{avg_sparsity_ig}{RESET_b}")

          # # Example: Compute feature interaction for Integrated Gradients
          # ig_interactions, ig_avg_interaction = compute_feature_interaction(attr_list, method="IntegratedGradients")
          # # Visualize average interaction matrix for DeepLift
          # visualize_feature_interaction(
          #     interaction_matrix=ig_avg_interaction,
          #     method="IntegratedGradients",
          #     title_suffix="(Average)"
          # )

          # Save attributions
          tmp__attributions_file_dir = os.path.join(save_path, f'{self.model_name}_attributions.npy')
          np.save(tmp__attributions_file_dir, attr_list)
          print(f"{GREEN}[+]{RESET} created IntegratedGradients file @ '{BLUE_b}{tmp__attributions_file_dir}{RESET_b}'.")

          # Save Metrics for Infidelity and 
          avg_infidelity = np.mean(infidelity_scores)
          std_infidelity = np.std(infidelity_scores)
          
          metric_vals = {
            'model_name': [self.model_name],
            'avg_entropy_ig': [avg_entropy_ig],
            'avg_infidelity': [avg_infidelity],
            'std_infidelity': [std_infidelity],
            'method': ["IntegratedGradients"]
          }
          df_metrics = pd.DataFrame(metric_vals)
          df_metrics.to_parquet(os.path.join(save_path, f"IntegratedGradient_metrics_table.parquet"))
          #avg_sensitivity = np.mean(sensitivity_scores)
          #std_sensitivity = np.std(infidelity_scores)
          metrics_file = os.path.join(save_path, f"IntegratedGradient_metrics.txt")
          with open(metrics_file, "w") as f:
              f.write(f"Average Infidelity: {avg_infidelity}\n")
              f.write(f"Std Infidelity: {std_infidelity}\n")
              #f.write(f"Average Sensitivity: {avg_sensitivity}\n")
              #f.write(f"Std Infidelity: {std_sensitivity}\n")
          print(f"{GREEN}[+]{RESET} Metrics for IntegratedGradient saved at {BLUE_b}{metrics_file}{RESET_b}")
          print(f"{GREEN}[+]{RESET} Average Infidelity: {BLUE_b}{avg_infidelity:}{RESET_b}")
          #print(f"{GREEN}[+]{RESET} Average Sensitivity: {BLUE_b}{avg_sensitivity:}{RESET_b}")

    def interpret_model_using_deeplift(self, data_loader, num_samples=10, save_visualizations=True, test_feature_mapping=None, perturb_std=None):
        self.eval()
        attr_list = []
        save_path = os.path.join(self.experiment_dir, "metrics") if save_visualizations else None
        os.makedirs(save_path, exist_ok=True)
        infidelity_scores = []
        if self.model_name == 'timemil' or self.model_name == 'timemil_extended':
            attention_weights = self.attention_weights.cpu().numpy()
            print("shape attention weights output of model: ", attention_weights.shape)

        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            x, y, lengths = batch
            x = x.to(self.device)
            print("Shape of x in batch: ", x.shape)
            y = y.to(self.device)
            print("Shape of y in batch: ", y.shape)
            lengths = lengths.to(self.device) if lengths is not None else None
            print("Shape of lengths in batch: ", lengths.shape)

            # Decode one-hot encoded labels into scalar indices
            y_indices = torch.argmax(y, dim=1)  # Convert one-hot to class indices
            target = y_indices[0].item()  # Get the first sample's target as a scalar

            # Dynamically adapt the forward method to return only logits
            original_forward = self.model.forward  # Backup the original forward method

            def perturb_fn(inputs):
                perturb_std_expanded = perturb_std.view(1, 1, -1).to(inputs.device)
                noise = torch.randn_like(inputs) * perturb_std_expanded
                perturbed_inputs = inputs + noise
                return noise, perturbed_inputs  # Swap the order here


            if self.model_name == "timemil"  or self.model_name == "timemil_extended":
                def adapted_forward(*inputs, **kwargs):
                    outputs = original_forward(*inputs, **kwargs)
                    if isinstance(outputs, tuple):  # If the model returns a tuple
                        return outputs[0]  # Return only logits
                    return outputs  # If already a tensor, return as-is
                additional_args = None
            elif self.model_name == "todynet":
                def adapted_forward(input_x):
                    # Adjust input shape as required by TodyNet
                    input_x = input_x.permute(0, 2, 1)  # [batch_size, num_nodes, seq_length]
                    input_x = input_x.unsqueeze(1)       # [batch_size, 1, num_nodes, seq_length]
                    logits, _, _ = original_forward(input_x)
                    return logits
                additional_args = None
            elif self.model_name == "lstm_classifier":
                def adapted_forward(input_x, lengths):
                    logits, _ = original_forward(input_x, lengths)
                    return logits
                additional_args = lengths            
            
            self.model.forward = adapted_forward  # Replace the model's forward method

            try:
                # Initialize DeepLift with the adapted model
                dl = DeepLift(self.model)

                print(f"Input shape before DeepLift: {x.shape}")
                print("Calculating attributions using DeepLift")
                attr = dl.attribute(
                    inputs=x,
                    target=target,
                    additional_forward_args=lengths if self.model_name == 'lstm_classifier' else None,
                )

                print("Target values", target)

                # Compute Infidelity
                infidelity_score = infidelity(adapted_forward, 
                                            perturb_fn, 
                                            x, 
                                            attr, 
                                            target=target,
                                            additional_forward_args=additional_args,  
                                            normalize=True
                )
                print(f"\n\nInfidelity score tensor shape: {infidelity_score.shape}, value: {infidelity_score}")
                infidelity_scores.extend(infidelity_score.cpu().tolist())  # Convert to list for easy handling
            

                inputs = x.cpu().detach().numpy()
                targets = y_indices.cpu().detach().numpy()
                attr = attr.cpu().detach().numpy()

                attr_list.append((inputs, attr, target))
                print("Finished calculating DeepLift values using Captum.")

                if i == 0:
                    channels_last_in = True

                    if x.shape[0] == self.input_dim:
                        channels_last_in = False
                    elif x.shape[1] == self.input_dim:
                        channels_last_in = True
                    else:
                        print(f"Warning: Unable to determine channels_last_in. Defaulting to True.")

                    print("targets: ", targets)
                    print("Value of channels_last_in:", channels_last_in)
                    print("Creating visualization of DeepLift attributions as Saliency Map using Captum.")
                    # Visualize attributions
                    if save_visualizations and i < 1:
                        visualize_attributions(attr=attr, 
                                              inputs=inputs, 
                                              targets=targets, 
                                              save_path=save_path, 
                                              num_samples=1, 
                                              channels_last=channels_last_in, 
                                              method_used="DeepLift", test_feature_mapping=test_feature_mapping)

            finally:
                # Restore the original forward method after attribution computation
                self.model.forward = original_forward

        # Compute global feature importance using Integrated Gradients attributions
        dl_global_importance = compute_global_feature_importance(attr_list, method="mean")
        np.save(os.path.join(self.experiment_dir, f"{self.model_name}_DeepLift_Global_Importance.npy"), dl_global_importance)
        print(f"Saved global feature importance for DeepLift at '{self.experiment_dir}'")
        visualize_feature_importance(dl_global_importance, method="DeepLift", aggregation="mean", test_feature_mapping=test_feature_mapping, save_path=self.experiment_dir)

        # Example usage for Integrated Gradients
        dl_entropies  = compute_entropy(attr_list)
        avg_entropy_dl = np.mean(dl_entropies)
        print(f"{GREEN}[+]{RESET}Average Entropy (DeepLift): {BLUE_b}{avg_entropy_dl}{RESET_b}")

        # Example: Compute sparsity for DeepLift
        dl_sparsities = compute_sparsity(attr_list, method="DeepLift", threshold=0.01)
        avg_sparsity_dl = np.mean(dl_sparsities)
        print(f"{GREEN}[+]{RESET}Average Sparsity (DeepLift): {BLUE_b}{avg_sparsity_dl}{RESET_b}")

        # # Example: Compute feature interaction for Integrated Gradients
        # dl_interactions, dl_avg_interaction = compute_feature_interaction(attr_list, method="DeepLift")
        # # Visualize average interaction matrix for DeepLift
        # visualize_feature_interaction(
        #     interaction_matrix=dl_avg_interaction,
        #     method="DeepLift",
        #     title_suffix="(Average)"
        # )

        # Save attributions
        tmp__attributions_file_dir = os.path.join(self.experiment_dir, f'{self.model_name}_DeepLift_attributions.npy')
        np.save(tmp__attributions_file_dir, attr_list)
        print(f"{GREEN}[+]{RESET} created DeepLift attributions file @ '{BLUE_b}{tmp__attributions_file_dir}{RESET_b}'.")

        # Save Metrics for Infidelity and 
        avg_infidelity = np.mean(infidelity_scores)
        std_infidelity = np.std(infidelity_scores)

        metric_vals = {
          'model_name': [self.model_name],
          'avg_entropy_ig': [avg_entropy_dl],
          'avg_infidelity': [avg_infidelity],
          'std_infidelity': [std_infidelity],
          'method': ["DeepLift"]
        }
        df_metrics = pd.DataFrame(metric_vals)
        df_metrics.to_parquet(os.path.join(save_path, f"DeepLift_metrics_table.parquet"))
        
        #avg_sensitivity = np.mean(sensitivity_scores)
        #std_sensitivity = np.std(infidelity_scores)
        metrics_file = os.path.join(save_path, f"DeepLift_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Average Infidelity: {avg_infidelity}\n")
            f.write(f"Std Infidelity: {std_infidelity}\n")
            #f.write(f"Average Sensitivity: {avg_sensitivity}\n")
            #f.write(f"Std Infidelity: {std_sensitivity}\n")
        print(f"{GREEN}[+]{RESET} Metrics for DeepLift saved at {BLUE_b}{metrics_file}{RESET_b}")
        print(f"{GREEN}[+]{RESET} Average Infidelity: {BLUE_b}{avg_infidelity:}{RESET_b}")
              

    
    def interpret_model_using_shapley_values(self, data_loader, num_samples=10, save_visualizations=True, test_feature_mapping=None):
          self.eval()
          attr_list = []
          save_path = os.path.join(self.experiment_dir, "visualizations") if save_visualizations else None

          if self.model_name == 'timemil'  or self.model_name == "timemil_extended":
              attention_weights = self.attention_weights.cpu().numpy()
              print("shape attention weights output of model: ", attention_weights.shape)
          # elif self.model_name == 'todynet':
          #     print("Length Intermediate outputs: ", len(self.intermediate_outputs))
          #     print("DataType Intermediate outputs: ", type(self.intermediate_outputs))
          #     print("Example Intermediate outputs: ", type(self.intermediate_outputs[0]))
          #     print("Example Intermediate outputs: ", self.intermediate_outputs[0])
          #     intermediate_outputs = self.intermediate_outputs.cpu().numpy()
          #     print("shape intermediate_outputs output of model: ", intermediate_outputs.shape)
          
          for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            x, y, lengths = batch
            x = x.to(self.device)
            print("Shape of x in batch: ", x.shape)
            y = y.to(self.device)
            print("Shape of y in batch: ", y.shape)
            lengths = lengths.to(self.device) if lengths is not None else None
            print("Shape of lengths in batch: ", lengths.shape)
            # target = y[0].item()  # Get the first sample's target
            
            # Decode one-hot encoded labels into scalar indices
            y_indices = torch.argmax(y, dim=1)  # Convert one-hot to class indices
            target = y_indices[0].item()  # Get the first sample's target as a scalar

            # Define a wrapper function based on the model
            if self.model_name == 'timemil' or self.model_name == "timemil_extended":
                def model_forward(input_x):
                    logits, _ = self.model(input_x)
                    return logits
                additional_args = None
            elif self.model_name == 'lstm_classifier':
                def model_forward(input_x, lengths):
                    logits, _ = self.model(input_x, lengths)
                    return logits
                additional_args = lengths
            elif self.model_name == 'todynet':
                def model_forward(input_x):
                    input_x = input_x.permute(0, 2, 1)  # [batch_size, num_nodes, seq_length]
                    input_x = input_x.unsqueeze(1) # [batch_size, 1, num_nodes, seq_length]
                    logits, _, _ = self.model(input_x)
                    return logits
                additional_args = None
            else:
                raise ValueError(f"Model {self.model_name} not supported for interpretability.")

            sv = ShapleyValues(model_forward)

            print(f"Input shape before SV: {x.shape}")
            print("Calculating ShapleyValues using Captum")
            attr = sv.attribute(
                inputs=x,
                target=target,
                additional_forward_args=additional_args,
                # return_convergence_delta=True,
                #n_steps=50,
                #internal_batch_size=x.shape[0]  # Set to batch size
            )
            
            inputs = x.cpu().detach().numpy()
            targets = y_indices.cpu().detach().numpy()            
            attr = attr.cpu().detach().numpy()
            
            # attr_list.append((x.cpu().numpy(), attr, target))
            attr_list.append((x.cpu().numpy(), attr, target))
            print("Finished calculating ShapleyValues using Captum.")
            print("attr shape:", attr.shape)
            print("inputs shape:", x.cpu().numpy().shape)
            #print("target shape:", target.shape)

            if i == 0:
                channels_last_in = True

                if x.shape[0] == self.input_dim:
                    channels_last_in=False
                elif x.shape[1] == self.input_dim:
                    channels_last_in=True
                else:
                    print(f"Warning: Unable to determine channels_last_in. Defaulting to True.")

                print("Value of channels_last_in:", channels_last_in)
                print("Creating visualization of ShapleyValues as Saliency Map using Captum.")
                print("test_feature_mapping:", test_feature_mapping)
                # Visualize attributions
                if save_visualizations and i < 1:
                    visualize_attributions(attr, 
                                          inputs, 
                                          targets, 
                                          save_path=save_path, 
                                          num_samples=1, 
                                          channels_last=channels_last_in, 
                                          method_used="ShapleyValues",
                                          test_feature_mapping=test_feature_mapping)
          
          # Compute global feature importance using Integrated Gradients attributions
          sv_global_importance = compute_global_feature_importance(attr_list, method="mean")
          np.save(os.path.join(self.experiment_dir, f"{self.model_name}_SV_Global_Importance.npy"), sv_global_importance)
          print(f"Saved global feature importance for ShapleyValues at '{self.experiment_dir}'")

          visualize_feature_importance(sv_global_importance, method="ShapleyValues", aggregation="mean", test_feature_mapping=test_feature_mapping, save_path=self.experiment_dir)

          # Example usage for Integrated Gradients
          sv_entropies = compute_entropy(attr_list)
          avg_entropy_sv = np.mean(sv_entropies)
          print(f"{GREEN}[+]{RESET}Average Entropy (ShapleyValues): {BLUE_b}{avg_entropy_sv}{RESET_b}")
          
          # Example: Compute sparsity for Integrated Gradients
          sv_sparsities = compute_sparsity(attr_list, method="ShapleyValues", threshold=1e-3)
          avg_sparsity_sv = np.mean(sv_sparsities)
          print(f"{GREEN}[+]{RESET}Average Sparsity (ShapleyValues): {BLUE_b}{avg_sparsity_sv}{RESET_b}")

          # # Example: Compute feature interaction for Integrated Gradients
          # ig_interactions, ig_avg_interaction = compute_feature_interaction(attr_list, method="IntegratedGradients")
          # # Visualize average interaction matrix for DeepLift
          # visualize_feature_interaction(
          #     interaction_matrix=ig_avg_interaction,
          #     method="IntegratedGradients",
          #     title_suffix="(Average)"
          # )

          # Save attributions
          tmp__attributions_file_dir = os.path.join(self.experiment_dir, f'{self.model_name}_ShapleyValues_attributions.npy')
          np.save(tmp__attributions_file_dir, attr_list)
          print(f"{GREEN}[+]{RESET} created ShapleyValues file @ '{BLUE_b}{tmp__attributions_file_dir}{RESET_b}'.")




    # def calculate_dataset_mean(self, data_loader, device):
    #     """
    #     Calculate the mean of the dataset for use as a baseline.
    
    #     Args:
    #         data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    #         device (torch.device): Device to ensure consistency.
    
    #     Returns:
    #         torch.Tensor: Mean tensor with shape matching the input features.
    #     """
    #     total_sum = None
    #     total_count = 0
    
    #     for batch in data_loader:
    #         x, _, _ = batch  # Assuming the data loader returns (inputs, labels, lengths)
    #         x = x.to(device)
    
    #         # Initialize total_sum if not already done
    #         if total_sum is None:
    #             total_sum = torch.zeros_like(x.sum(dim=0))
    
    #         total_sum += x.sum(dim=0)  # Sum across the batch
    #         total_count += x.shape[0]  # Count total samples
    
    #     dataset_mean = total_sum / total_count  # Mean over all batches
    #     return dataset_mean

    # def interpret_model_using_deeplift_shap(self, data_loader, num_samples=10, save_visualizations=True):
    #     self.eval()
    #     attr_list = []
    #     save_path = os.path.join(self.experiment_dir, "visualizations") if save_visualizations else None

    #     # Calculate dataset mean for baseline
    #     #dataset_mean = self.calculate_dataset_mean(data_loader, self.device)
    #     #print(f"Calculated dataset mean with shape: {dataset_mean.shape}")

    #     if self.model_name == 'timemil':
    #         attention_weights = self.attention_weights.cpu().numpy()
    #         print("shape attention weights output of model: ", attention_weights.shape)

    #     for i, batch in enumerate(data_loader):
    #         if i >= num_samples:
    #             break
    #         x, y, lengths = batch
    #         x = x.to(self.device)
    #         print("Shape of x in batch: ", x.shape)
    #         y = y.to(self.device)
    #         print("Shape of y in batch: ", y.shape)
    #         lengths = lengths.to(self.device) if lengths is not None else None
    #         print("Shape of lengths in batch: ", lengths.shape)

    #         # Decode one-hot encoded labels into scalar indices
    #         y_indices = torch.argmax(y, dim=1)  # Convert one-hot to class indices
    #         target = y_indices[0].item()  # Get the first sample's target as a scalar

    #         # Define baseline (e.g., tensor of zeros or dataset mean)
    #         baseline = torch.zeros_like(x).to(self.device)

    #         # Dynamically adapt the forward method to return only logits
    #         original_forward = self.model.forward  # Backup the original forward method

    #         if self.model_name == "timemil":
    #             def adapted_forward(*inputs, **kwargs):
    #                 outputs = original_forward(*inputs, **kwargs)
    #                 if isinstance(outputs, tuple):  # If the model returns a tuple
    #                     return outputs[0]  # Return only logits
    #                 return outputs  # If already a tensor, return as-is
    #             additional_args = None
    #         elif self.model_name == "todynet":
    #             def adapted_forward(input_x):
    #                 # Adjust input shape as required by TodyNet
    #                 input_x = input_x.permute(0, 2, 1)  # [batch_size, num_nodes, seq_length]
    #                 input_x = input_x.unsqueeze(1)       # [batch_size, 1, num_nodes, seq_length]
    #                 logits, _, _ = original_forward(input_x)
    #                 return logits
    #             additional_args = None
    #         elif self.model_name == "lstm_classifier":
    #             def adapted_forward(input_x, lengths):
    #                 logits, _ = original_forward(input_x, lengths)
    #                 return logits
    #             additional_args = (lengths,)             
            
    #         self.model.forward = adapted_forward  # Replace the model's forward method

    #         try:
    #             # Initialize DeepLift with the adapted model
    #             dl_shap = DeepLiftShap(self.model)
                
    #             #expanded_baseline = dataset_mean.unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)

    #             print(f"Input shape before DeepLiftShap: {x.shape}")
    #             print("Calculating attributions using DeepLiftShap")
    #             attr = dl_shap.attribute(
    #                 inputs=x,
    #                 baselines=baseline,  # Use dataset mean as baseline
    #                 target=target,
    #                 additional_forward_args=lengths if self.model_name == 'lstm_classifier' else None,
    #             )

    #             inputs = x.cpu().detach().numpy()
    #             targets = y_indices.cpu().detach().numpy()
    #             attr = attr.cpu().detach().numpy()

    #             attr_list.append((inputs, attr, target))
    #             print("Finished calculating DeepLiftShap values using Captum.")

    #             if i == 0:
    #                 channels_last_in = True

    #                 if x.shape[0] == self.input_dim:
    #                     channels_last_in = False
    #                 elif x.shape[1] == self.input_dim:
    #                     channels_last_in = True
    #                 else:
    #                     print(f"Warning: Unable to determine channels_last_in. Defaulting to True.")

    #                 print("Value of channels_last_in:", channels_last_in)
    #                 print("Creating visualization of DeepLiftShap attributions as Saliency Map using Captum.")
    #                 # Visualize attributions
    #                 if save_visualizations and i < 1:
    #                     visualize_attributions(attr, 
    #                                           inputs, 
    #                                           targets, 
    #                                           save_path=save_path, 
    #                                           num_samples=1, 
    #                                           channels_last=channels_last_in, 
    #                                           method_used="DeepLiftShap")

    #         finally:
    #             # Restore the original forward method after attribution computation
    #             self.model.forward = original_forward

    #     # Save attributions
    #     tmp__attributions_file_dir = os.path.join(self.experiment_dir, f'{self.model_name}_DeepLiftShap_attributions.npy')
    #     np.save(tmp__attributions_file_dir, attr_list)
    #     print(f"{GREEN}[+]{RESET} created DeepLift attributions file @ '{BLUE_b}{tmp__attributions_file_dir}{RESET_b}'.")


    def visualize_attention(self, data_loader, num_samples=1, save_path=None):
        """
        Visualize attention weights for the TimeMIL model.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            num_samples (int): Number of samples to visualize.
            save_path (str, optional): Path to save visualizations. If None, visualizations are displayed interactively.
        """
        self.eval()
        save_path = save_path or os.path.join(self.experiment_dir, "attention_visualizations")
        os.makedirs(save_path, exist_ok=True)

        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break

            x, y, lengths = batch
            x = x.to(self.device)
            lengths = lengths.to(self.device) if lengths is not None else None


            with torch.no_grad():
                if self.model_name == "timemil" or self.model_name == "timemil_extended":
                    logits, attention_weights = self.model(x)
                    attention_weights = attention_weights.cpu().numpy()
                    x_np = x.cpu().numpy()

                    for j in range(min(x_np.shape[0], num_samples)):
                        instance_data = x_np[j]  # Shape: (seq_len, input_dim)
                        instance_attention = attention_weights[j]  # Shape: (seq_len, )

                        # Transpose data to match expected input format for visualization
                        if instance_data.shape[0] != instance_attention.shape[0]:
                            instance_data = instance_data.T  # Ensure (seq_len, input_dim)

                        title = f"Attention Visualization for Sample {j}"
                        output_file = os.path.join(save_path, f"attention_sample_{j}.png")

                        # Call visualization function
                        fig, _ = visualize_attention_weights(
                            attention_weights=instance_attention,
                            data=instance_data,
                            method="overlay_individual",
                            title=title,
                            fig_size=(12, 6),
                        )

                        # Save or show the figure
                        if save_path:
                            fig.savefig(output_file)
                            plt.close(fig)
                            print(f"{GREEN}[+]{RESET} Attention visualization saved to {BLUE_b}{output_file}{RESET_b}")
                        else:
                            plt.show()
                else:
                    raise ValueError(f"Visualization is only supported for 'timemil' model, got {self.model_name}.")


    def build_model(self):
        if self.model_name == 'lstm':
            
            self.model = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True
            )
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        elif self.model_name == 'lstm_classifier':
            
            # Initialize LSTMClassifier
            self.model = LSTMClassifier(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )

        elif self.model_name == 'todynet':
            
            # Remove duplicate keys from todynet_params
            todynet_params = self.todynet_params.copy()

            # Remove duplicates
            for key in ['input_dim', 'seq_length', 'num_nodes', 'num_classes']:
                if key in todynet_params:
                    print(f"Removing {key} from todynet_params to avoid duplicates")
                    todynet_params.pop(key)

            print(f"Final todynet_params being passed: {todynet_params}")

            # Pass the modified todynet_params
            self.model = TodyNetClassifier(
                input_dim=todynet_params["in_dim"],
                seq_length=self.max_seq_len,
                num_nodes=self.input_dim,  # Or set appropriately
                num_classes=self.num_classes,
                **todynet_params  # Pass additional parameters
            )
        elif self.model_name == 'timemil':
            
            # Initialize TimeMIL model
            self.model = TimeMIL(
                in_features=self.input_dim,
                n_classes=self.num_classes,
                mDim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                dropout=self.dropout
            )
        elif self.model_name == 'timemil_extended':
            
            # Initialize TimeMIL model
            self.model = TimeMIL_extended(
                in_features=self.input_dim,
                n_classes=self.num_classes,
                mDim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                dropout=self.dropout
            )

        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def on_fit_start(self):
        print("\t{YELLOW}[->]{RESET}Resetting training and validation metrics to ensure correct value allocation.")
        # Reset training metrics
        self.train_accuracy.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_auc.reset()

        # Reset validation metrics
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auc.reset()   

    def on_test_start(self):
        print("\t{YELLOW}[->]{RESET}Resetting test metrics to ensure correct value allocation.")
        # Reset test metrics
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_auc.reset()

        # Reset test predictions and labels
        self.test_preds = []
        self.test_labels = []

    def on_train_start(self):
        if self.use_class_weights and self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, lengths):
        if self.model_name == 'lstm':
            # Pack sequences for LSTM
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hn, cn) = self.model(packed_input)
            # Use the last hidden state
            hn = hn[-1]  # Get the last layer
            logits = self.classifier(hn)
            probs = F.softmax(logits, dim=1)
        elif self.model_name == 'lstm_classifier':
            logits, probs = self.model(x, lengths)
            return logits, probs
        elif self.model_name == 'timemil' or self.model_name == "timemil_extended":
            # print(f"Shape x before passing to model: {x.shape}")  # Before transpose
            logits, attention_weights = self.model(x)
            probs = F.softmax(logits, dim=1)
            self.attention_weights = attention_weights  # Store for later use
            
            # Debugging shapes
            #print(f"Shape of input (x): {x.shape}")
            #print(f"Shape of attention_weights: {attention_weights.shape}")

            return logits, probs
        elif self.model_name == 'todynet':
            # Adjust input shape to match TodyNet's expectations
            #print(f"Shape of x before permutation: {x.shape}")
            #print(f"batch_size: {x.shape[0]}")
            #print(f"num_nodes: {x.shape[2]}")
            #print(f"seq_length: {x.shape[1]}")
            x = x.permute(0, 2, 1)  # [batch_size, num_nodes, seq_length]
            #print(f"Shape of x after permutation: {x.shape}")
            x = x.unsqueeze(1)       # [batch_size, 1, num_nodes, seq_length]
            #print(f"Shape of x after adding new dimension 1 at position 1 in array: {x.shape}")
            logits, probs, intermediate_outputs = self.model(x)
            self.intermediate_outputs = intermediate_outputs
            return logits, probs
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
    
    def configure_optimizers(self):
        # Instantiate optimizer
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")
        
        # Wrap with Lookahead if required
        if self.use_lookahead:
            optimizer = Lookahead(optimizer, **self.lookahead_params)
        
        # Set up scheduler
        if self.scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_params)
        elif self.scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        elif self.scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        elif self.scheduler_name is None:
            scheduler = None
        else:
            raise ValueError(f"Scheduler {self.scheduler_name} not supported.")
        
        # Return optimizer (and scheduler if provided)
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss' if self.scheduler_name == 'reduce_on_plateau' else None
            }
        else:
            return optimizer
    
    
    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits, probs = self(x, lengths)
        loss = self.criterion(logits, y)

        preds = torch.argmax(probs, dim=1) # changed from logits to probs
        y_indices = torch.argmax(y, dim=1)  # Convert one-hot encoded labels to indices

        self.train_accuracy(preds, y_indices)
        self.train_f1(preds, y_indices)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1)

        self.train_precision(preds, y_indices)
        self.train_recall(preds, y_indices)
        # self.train_auc(logits.softmax(dim=-1), y_indices)
        self.train_auc(probs, y_indices)
        self.log('train_precision', self.train_precision)
        self.log('train_recall', self.train_recall)
        self.log('train_auc', self.train_auc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits, probs = self(x, lengths)
        loss = self.criterion(logits, y)
    
        preds = torch.argmax(probs, dim=1) # changed from logits to probs
        y_indices = torch.argmax(y, dim=1)  # Convert one-hot encoded labels to indices
    
        self.val_accuracy(preds, y_indices)
        self.val_f1(preds, y_indices)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1)
    
        self.val_precision(preds, y_indices)
        self.val_recall(preds, y_indices)
        # self.val_auc(logits.softmax(dim=-1), y_indices)
        self.val_auc(probs, y_indices)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_auc', self.val_auc)
    
        return loss

    
    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits, probs = self(x, lengths)
        loss = self.criterion(logits, y)
    
        preds = torch.argmax(probs, dim=1) # changed from logits to probs
        y_indices = torch.argmax(y, dim=1)  # Convert one-hot encoded labels to indices
    
        print(f"Shape of logits: {logits.shape}")
        print(f"Shape of probs: {probs.shape}")
        print(f"Shape of preds: {preds.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Shape of y_indices: {y_indices.shape}")

        self.test_accuracy(preds, y_indices)
        self.test_f1(preds, y_indices)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy)
        self.log('test_f1', self.test_f1)
    
        self.test_precision(preds, y_indices)
        self.test_recall(preds, y_indices)
        # self.test_auc(logits.softmax(dim=-1), y_indices)
        self.test_auc(probs, y_indices)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        self.log('test_auc', self.test_auc)
    
        # Save predictions and labels
        self.test_preds.append(preds.cpu())
        self.test_probs.append(probs.cpu())
        self.test_labels.append(y_indices.cpu())
    
        return loss
    
    # # Old method (working)
    # def on_test_epoch_end(self):
    #     # Concatenate predictions and labels
    #     preds = torch.cat(self.test_preds)
    #     labels = torch.cat(self.test_labels)

    #     # Validate `probs` shape
    #     print(f" Shape of labels: {labels.shape}")

    #     # Save results
    #     results_df = pd.DataFrame({'predictions': preds.numpy(), 'labels': labels.numpy()})
    #     results_file = os.path.join(self.experiment_dir, 'test_results.csv')
    #     os.makedirs(os.path.dirname(results_file), exist_ok=True)
    #     try:
    #         results_df.to_csv(results_file, index=False)
    #         print(f"{GREEN}[+]{RESET}Test results saved to {BLUE_b}{results_file}{RESET_b}")
    #     except Exception as e:
    #         print(f"{RED}[-]{RESET}Error saving test results: {e}")

    # def on_test_epoch_end(self):
    #     # Concatenate predictions and labels
    #     preds = torch.cat(self.test_preds)
    #     labels = torch.cat(self.test_labels)
    #     probs = torch.cat(self.test_probs)
    
    #     # Validate shapes
    #     print(f"Shape of labels: {labels.shape}")
    #     print(f"Shape of predictions: {preds.shape}")
    #     print(f"Shape of probabilities: {probs.shape}")
    
    #     # Save results with logits
    #     results_df = pd.DataFrame({'predictions': preds.numpy(), 'labels': labels.numpy()})
    #     results_file = os.path.join(self.experiment_dir, 'test_results_logits.csv')
    #     os.makedirs(os.path.dirname(results_file), exist_ok=True)
    #     try:
    #         results_df.to_csv(results_file, index=False)
    #         print(f"{GREEN}[+]{RESET}Test results saved to {BLUE_b}{results_file}{RESET_b}")
    #     except Exception as e:
    #         print(f"{RED}[-]{RESET}Error saving test results: {e}")
    
    #     # Save probabilities and labels to a DataFrame
    #     try:
    #         results_df_probs = pd.DataFrame({
    #             f'prob_class_{i}': probs[:, i].cpu().numpy() for i in range(probs.shape[1])
    #         })
    #         results_df_probs['labels'] = labels.cpu().numpy()
    #         results_file_probs = os.path.join(self.experiment_dir, 'test_results_probs.csv')
    #         os.makedirs(os.path.dirname(results_file_probs), exist_ok=True)
    #         results_df_probs.to_csv(results_file_probs, index=False)
    #         print(f"{GREEN}[+]{RESET}Test results saved to {BLUE_b}{results_file_probs}{RESET_b}")
    #     except Exception as e:
    #         print(f"{RED}[-]{RESET}Error saving test results: {e}")
    
    #     # Compute ROC and AUC
    #     n_classes = self.num_classes
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    
    #     # Convert labels to one-hot encoding
    #     labels_one_hot = label_binarize(labels.cpu().numpy(), classes=list(range(n_classes)))
    
    #     for i in range(n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], probs[:, i].cpu().numpy())
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    
    #     # Compute micro-average ROC curve and ROC area
    #     fpr["micro"], tpr["micro"], _ = roc_curve(labels_one_hot.ravel(), probs.cpu().numpy().ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    #     # Plot ROC curve for each class
    #     plt.figure(figsize=(10, 8))
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    #     for i, color in zip(range(n_classes), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #                  label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    
    #     # Plot micro-average ROC curve
    #     plt.plot(fpr["micro"], tpr["micro"],
    #              label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
    #              color='deeppink', linestyle=':', linewidth=4)
    
    #     # Plot diagonal line for random guess
    #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     roc_plot_file = os.path.join(self.experiment_dir, 'roc_curve.png')
    #     plt.savefig(roc_plot_file)
    #     plt.close()
    #     print(f"{GREEN}[+]{RESET}ROC curve saved to {BLUE_b}{roc_plot_file}{RESET_b}")

    # new method here for testing
    def on_test_epoch_end(self):
        # Concatenate predictions and labels
        preds = torch.cat(self.test_preds)
        labels = torch.cat(self.test_labels)
        probs = torch.cat(self.test_probs)
        
        # Convert one-hot encoded labels to class indices if they are not already
        if labels.ndim == 2 and labels.size(1) > 1:  # One-hot encoded
            labels = labels.argmax(dim=1)
        
        # Validate shapes
        print(f"Shape of labels: {labels.shape}")
        print(f"Shape of predictions: {preds.shape}")
        print(f"Shape of probabilities: {probs.shape}")
        
        # Save results with logits
        results_df = pd.DataFrame({'predictions': preds.numpy(), 'labels': labels.numpy()})
        results_file = os.path.join(self.experiment_dir, 'test_results_logits.csv')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        try:
            results_df.to_csv(results_file, index=False)
            print(f"{GREEN}[+]{RESET}Test results saved to {BLUE_b}{results_file}{RESET_b}")
        except Exception as e:
            print(f"{RED}[-]{RESET}Error saving test results: {e}")
        
        # Save probabilities and labels to a DataFrame
        try:
            results_df_probs = pd.DataFrame({
                f'prob_class_{i}': probs[:, i].cpu().numpy() for i in range(probs.shape[1])
            })
            results_df_probs['labels'] = labels.cpu().numpy()
            results_file_probs = os.path.join(self.experiment_dir, 'test_results_probs.csv')
            os.makedirs(os.path.dirname(results_file_probs), exist_ok=True)
            results_df_probs.to_csv(results_file_probs, index=False)
            print(f"{GREEN}[+]{RESET}Test results saved to {BLUE_b}{results_file_probs}{RESET_b}")
        except Exception as e:
            print(f"{RED}[-]{RESET}Error saving test results: {e}")
        
        # Compute ROC and AUC
        n_classes = self.num_classes
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        if n_classes == 2:  # Binary classification case
            fpr[1], tpr[1], _ = roc_curve(labels.cpu().numpy(), probs[:, 1].cpu().numpy())
            roc_auc[1] = auc(fpr[1], tpr[1])
            fpr[0] = 1 - fpr[1]  # Assuming binary probabilities complement each other
            tpr[0] = 1 - tpr[1]
            roc_auc[0] = 1 - roc_auc[1]
        else:  # Multiclass classification case
            # Convert labels to one-hot encoding
            labels_one_hot = label_binarize(labels.cpu().numpy(), classes=list(range(n_classes)))
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], probs[:, i].cpu().numpy())
                roc_auc[i] = auc(fpr[i], tpr[i])
        
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(labels_one_hot.ravel(), probs.cpu().numpy().ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        if n_classes == 2:  # Binary classification case
            plt.plot(fpr[1], tpr[1], color='blue', lw=2, label=f'ROC curve (area = {roc_auc[1]:0.2f})')
        else:  # Multiclass classification case
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
            # Plot micro-average ROC curve
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                     color='deeppink', linestyle=':', linewidth=4)
        
        # Plot diagonal line for random guess
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_plot_file = os.path.join(self.experiment_dir, 'roc_curve.png')
        plt.savefig(roc_plot_file)
        plt.close()
        print(f"{GREEN}[+]{RESET}ROC curve saved to {BLUE_b}{roc_plot_file}{RESET_b}")
    