from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
import pandas as pd

from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

class EnergyForecastingModel:
    """
    LSTM-based time series forecasting model for energy consumption.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3, batch_size=200, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.max_grad_norm = 5.0

        self.model = xLSTMLarge(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_dataloaders(self, X, Y, cols_cat, split_ratio=0.8):
        split_idx = int(split_ratio * len(X))

        X_main = X.drop(cols_cat, axis=1)
        X_train = self.scaler_X.fit_transform(X_main.iloc[:split_idx])
        X_val = self.scaler_X.transform(X_main.iloc[split_idx:])

        Y_train = self.scaler_Y.fit_transform(Y.iloc[:split_idx])
        Y_val = self.scaler_Y.transform(Y.iloc[split_idx:])

        X_train_cat = X[cols_cat].iloc[:split_idx]
        X_val_cat = X[cols_cat].iloc[split_idx:]

        X_train = np.concatenate([X_train, X_train_cat], axis=1)
        X_val = np.concatenate([X_val, X_val_cat], axis=1)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(Y_train, dtype=torch.float32)),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                              torch.tensor(Y_val, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self, X, Y, cols_cat, num_epochs=500, patience=25, seed=1, model_path="best_lstm_model.pth"):
        self._set_seed(seed)
        train_loader, val_loader = self._create_dataloaders(X, Y, cols_cat)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for X_batch, Y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch.unsqueeze(1))
                loss = self.criterion(outputs, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    outputs = self.model(X_batch.unsqueeze(1))
                    loss = self.criterion(outputs, Y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break

        self.model.load_state_dict(torch.load(model_path))
        print("Best model loaded.")
        self._plot_loss(train_losses, val_losses)

    def predict(self, test_data, seq_length, cols_cat, target_columns):
        predictions = pd.DataFrame(index=test_data.index, columns=target_columns)

        for i, timestamp in enumerate(test_data.index[-len(predictions):]):
            start_idx = test_data.index.get_loc(timestamp) - seq_length
            end_idx = test_data.index.get_loc(timestamp)
            input_sequence = test_data.iloc[start_idx:end_idx]

            input_sequence_scaled = self.scaler_X.transform(input_sequence.drop(cols_cat, axis=1))
            input_sequence_scaled = np.concatenate([input_sequence_scaled, input_sequence[cols_cat]], axis=1)
            input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                pred = self.model(input_tensor).squeeze(0).numpy()
                pred_original = self.scaler_Y.inverse_transform(pred.reshape(1, -1))[0]

            predictions.loc[timestamp] = pred_original
            test_data.loc[timestamp, target_columns] = pred_original

        return predictions
