'''
Author: Adrian Pal
Summary: This class provides utility methods for evaluating and visualizing the performance of machine learning models.
It includes methods for plotting training history, saving performance metrics, generating and saving plots, and recording training time.
'''

import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class ModelPerformanceUtil:
    def __init__(self, model_name, image_dir="Performance_Images"):
        """
        Initialize the ModelPerformanceUtil with model name and image directory.

        :param model_name: Name of the model.
        :param image_dir: Directory to save performance images.
        """
        self.model_name = model_name
        self.image_dir = image_dir
        self.history = None

        # Create the directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)

    def plot_training_history(self, history):
        """
        Plot the training history (accuracy and loss) and save the plots.

        :param history: Keras History object containing training history.
        :return: A plot of the training over a different amount of epochs.
        """
        self.history = history
        if self.history is not None:
            # Create a new figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot training & validation accuracy values
            ax1.plot(self.history.history['accuracy'], label='Train')
            ax1.plot(self.history.history['val_accuracy'], label='Validation')
            ax1.set_title(f'{self.model_name} Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend(loc='upper left')

            # Plot training & validation loss values
            ax2.plot(self.history.history['loss'], label='Train')
            ax2.plot(self.history.history['val_loss'], label='Validation')
            ax2.set_title(f'{self.model_name} Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper left')

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.image_dir, f'{self.model_name}_training_plots.png'))
            plt.close()

    def save_performance_metrics(self, metrics_dict):
        """
        Save the performance metrics to a text file.

        :param metrics_dict: Dictionary containing performance metrics.
        :return: A text file returning the performace metrics.
        """
        with open(os.path.join(self.image_dir, f'{self.model_name}_performance_metrics.txt'), 'w') as f:
            f.write(f'{self.model_name} Performance Metrics:\n')
            for metric_name, metric_value in metrics_dict.items():
                f.write(f'{metric_name}: {metric_value}\n')

    def generate_and_save_plots(self, history, y_true, y_pred):
        """
        Generate and save plots (training history) and save performance metrics.

        :param history: Keras History object containing training history.
        :param y_true:  True labels.
        :param y_pred: Predicted labels.
        :return: Images of the training plots.
        """
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

    def record_training_time(self, start_time, end_time):
        """
        Record the training time and save it to the performance metrics file.

        :param start_time: Start time of training.
        :param end_time: End time of training.
        :return: Time taken to train the model, text appended in the text file.
        """
        # Calculate the training time
        training_time = end_time - start_time

        # Convert to minutes if greater than threshold
        if training_time > 60:
            training_time /= 60
            time_unit = "minutes"
        else:
            time_unit = "seconds"

        # Write the training time to the performance metrics file
        with open(os.path.join(self.image_dir, f'{self.model_name}_performance_metrics.txt'), 'a') as f:
            f.write(f'Time taken: {training_time:.2f} {time_unit}\n')
