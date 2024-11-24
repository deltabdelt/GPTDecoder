from typing import Dict, List

import torch
from loguru import logger
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from model import GPTDecoder


class Evaluate:
    def __init__(self, model_params: Dict, best_model_path: str):
        self.model = GPTDecoder(model_params)
        self.best_model_path = best_model_path
        self.device = model_params.get('device')

        self.load_best_model()

    def load_best_model(self):
        """
        Loads the best model from a checkpoint and prepares it for evaluation.
        """
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        logger.info(f"Best model loaded from {self.best_model_path}")

    def evaluate_model(self, val_loader: DataLoader):
        """
        Evaluates the model on the validation dataset and collects predictions and labels.

        Args:
            model: The trained model.
            val_loader: DataLoader for the validation dataset.
            device: The device to use (CPU/GPU).

        Returns:
            y_true: Ground truth labels.
            y_pred: Predicted labels by the model.
        """
        y_true = []
        y_pred = []

        with torch.no_grad():
            for Xb, Yb, mask in val_loader:
                # Move data to the same device as the model
                Xb, Yb, mask = Xb.to(self.device), Yb.to(self.device), mask.to(self.device)

                # Get predictions from the model
                logits, _ = self.model(Xb, Yb, mask)
                predictions = torch.argmax(logits, dim=1)  # Predicted class

                # Append predictions and true labels
                y_true.extend(Yb.cpu().numpy())  # Convert to numpy for sklearn
                y_pred.extend(predictions.cpu().numpy())

        return y_true, y_pred
    
    def generate_classification_report(self, y_true: List, y_pred: List, label_mapping: Dict):
        """
        Generates and prints a classification report.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            label_mapping: Dictionary mapping label indices to label names.
        """
        # Convert label indices to label names
        target_names = [label_mapping[idx] for idx in range(len(label_mapping))]
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        
        return report
    
    def evaluate(self, val_loader: DataLoader, label_mapping: Dict):

        y_true, y_pred = self.evaluate_model(val_loader)

        report = self.generate_classification_report(y_true, y_pred, label_mapping)

        return report
    
