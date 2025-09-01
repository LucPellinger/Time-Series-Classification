# lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 dropout=0.0, bidirectional=False):
        """
        LSTM-based classifier for time series data.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes.
            dropout (float, optional): Dropout probability between LSTM layers. Default is 0.0.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default is False.
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Define the output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x, lengths):
        """
        Forward pass of the LSTM classifier.

        Args:
            x (torch.Tensor): Padded input sequences of shape (batch_size, seq_length, input_dim).
            lengths (torch.Tensor): Original lengths of sequences before padding.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Pack the sequences for efficient processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through the LSTM layers
        packed_output, (hn, cn) = self.lstm(packed_input)

        # hn shape: (num_layers * num_directions, batch_size, hidden_dim)
        # We need to extract the appropriate hidden state(s) to use for classification

        # If bidirectional, concatenate the last hidden state from both directions
        if self.bidirectional:
            # Reshape hn to (num_layers, num_directions, batch_size, hidden_dim)
            hn = hn.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
            # Select the last layer's forward and backward hidden states
            hn_forward = hn[-1, 0, :, :]  # Shape: (batch_size, hidden_dim)
            hn_backward = hn[-1, 1, :, :]  # Shape: (batch_size, hidden_dim)
            # Concatenate forward and backward hidden states
            hn = torch.cat((hn_forward, hn_backward), dim=1)  # Shape: (batch_size, hidden_dim * 2)
        else:
            # Use the last layer's hidden state
            hn = hn[-1, :, :]  # Shape: (batch_size, hidden_dim)

        # Pass the hidden state through the fully connected layer
        logits = self.fc(hn)  # Shape: (batch_size, num_classes)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        return logits, probs # Return logits for loss computation and interpretability
