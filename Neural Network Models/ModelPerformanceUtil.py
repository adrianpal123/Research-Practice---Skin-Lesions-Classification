import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class ModelPerformanceUtil:
    def __init__(self, model_name, image_dir="Performance_Images"):
        self.model_name = model_name
        self.image_dir = image_dir
        self.history = None

        # Create the directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)

    def plot_training_history(self, history):
        self.history = history
        if self.history is not None:
            # Plot training & validation accuracy values
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title(f'{self.model_name} Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(self.image_dir, f'{self.model_name}_accuracy_plot.png'))
            plt.close()

            # Plot training & validation loss values
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title(f'{self.model_name} Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(self.image_dir, f'{self.model_name}_loss_plot.png'))
            plt.close()

    def save_performance_metrics(self, metrics_dict):
        with open(os.path.join(self.image_dir, f'{self.model_name}_performance_metrics.txt'), 'w') as f:
            f.write(f'{self.model_name} Performance Metrics:\n')
            for metric_name, metric_value in metrics_dict.items():
                f.write(f'{metric_name}: {metric_value}\n')

    def generate_and_save_plots(self, history, y_true, y_pred):
        self.plot_training_history(history)

        # Compute evaluation metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        # Put the computed metrics into a dictionary
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        # Save performance metrics
        self.save_performance_metrics(metrics_dict)
