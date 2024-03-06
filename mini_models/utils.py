import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

def aggregate_and_compare_predictions(df, pred_column, true_column,agg_df_true='y_true',agg_df_pred='y_pred', id_column='id'):
    """
    Aggregates predictions and true labels by ID.

    Parameters:
    - df: DataFrame containing the predictions, true labels, and IDs.
    - pred_column: The name of the column containing model predictions.
    - true_column: The name of the column containing true labels.
    - id_column: The name of the column containing unique identifiers for aggregation.

    Returns:
    - agg_df: DataFrame with aggregated true labels and predictions.
    """
    # Drop rows where either prediction or true label is NaN
    comparison_df = df.dropna(subset=[pred_column, true_column])
    
    # Aggregate predictions and true labels
    agg_pred = comparison_df.groupby(id_column)[pred_column].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    agg_true = comparison_df.groupby(id_column)[true_column].agg(lambda x: x.iloc[0])
    
    # Create a DataFrame from the aggregated data
    agg_df = pd.DataFrame({agg_df_true: agg_true, agg_df_pred: agg_pred}).dropna()
    

    # Para pasar de IN OUT a numeros
    # Encode labels if they are categorical
    # if agg_df[agg_df_true].dtype == object or agg_df[agg_df_pred].dtype == object:
    #     le = LabelEncoder()
    #     agg_df[agg_df_true] = le.fit_transform(agg_df[agg_df_true])
    #     agg_df[agg_df_pred] = le.transform(agg_df[agg_df_pred])
    
    return agg_df

def plot_confusion_matrix(agg_df, true_label_col='y_true', pred_label_col='y_pred'):
    """
    Plots a confusion matrix based on aggregated true labels and predictions.

    Parameters:
    - agg_df: DataFrame with aggregated true labels and predictions.
    - true_label_col: Column name for the true label in agg_df.
    - pred_label_col: Column name for the predicted label in agg_df.
    """
    # Generate the confusion matrix
    cm = confusion_matrix(agg_df[true_label_col], agg_df[pred_label_col], labels=np.unique(agg_df[true_label_col]))
    
    # Display the confusion matrix
    cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(agg_df[true_label_col]))
    cmd.plot(cmap="Blues")
    cmd.ax_.set(xlabel='Predicted labels', ylabel='True labels', title='Confusion Matrix')
    plt.show()

# De los datos que nivel de confianza existe
def calculate_confidence_distribution(comparison_df, id_column=None, label_direction_column='', model_label_direction_column='', model_label_direction_conf_column=''):
    """
    Calculates and displays the distribution of confidence scores for true positive predictions.

    Parameters:
    - comparison_df: DataFrame containing the comparison data.
    - id_column: The name of the column containing unique identifiers.
    - label_direction_column: The name of the column containing true labels.
    - model_label_direction_column: The name of the column containing model predictions.
    - model_label_direction_conf_column: The name of the column containing model prediction confidences.

    Returns:
    - A distribution of confidence scores for true positive predictions.
    """
    # Middle prediction per ID for true positives
    if id_column is not None:
        comparison_df = comparison_df.groupby(id_column).apply(lambda x: x.iloc[(len(x) - 1) // 2])
    
    # Define bins for the confidence scores
    bins = np.arange(0.5, 1.05, 0.05)
    labels = [f"{i:.2f}-{i+0.05:.2f}" for i in bins[:-1]]  # Labels for the intervals
    
    # Bin the data
    comparison_df['conf_interval'] = pd.cut(
        comparison_df[model_label_direction_conf_column],
        bins=bins,
        labels=labels,
        right=False
    )
    
    # Count the occurrences in each bin
    distribution = comparison_df['conf_interval'].value_counts().sort_index()
    
    # Display the distribution
    return distribution

def plot_false_positives(agg_df, comparison_df,agg_df_true='y_true',agg_df_pred='y_pred', img_name_col='img_name', true_label_col='y_true', pred_label_col='y_pred', conf_col='model_label_direction_conf', base_img_path='', save_path='logs/false_positive_images.png', nrows=2):
    """
    Plots false positive images based on aggregated data.

    Parameters:
    - agg_df: DataFrame with aggregated data containing true and predicted labels.
    - comparison_df: Original comparison DataFrame with detailed data.
    - img_name_col: Column name containing the image filenames.
    - true_label_col: Column name for the true label.
    - pred_label_col: Column name for the predicted label.
    - conf_col: Column name for the model confidence.
    - base_img_path: Base path to the image directory.
    - save_path: Path to save the plotted figure.
    - nrows: Number of rows in the plot grid.
    """
    # Identify false positives at the ID level based on the aggregated data
    false_positive_ids = agg_df[(agg_df[agg_df_true] == 'IN') & (agg_df[agg_df_pred] == 'OUT')].index

    # Filter the original comparison_df to get rows that match false positive IDs
    false_positives_df = comparison_df[comparison_df['id'].isin(false_positive_ids)]
    false_positives_df = false_positives_df.dropna(subset=[img_name_col])

    # Group by ID and select the middle image for each group
    middle_images = false_positives_df.groupby('id').apply(lambda x: x.iloc[(len(x) - 1) // 2])
    false_positives_df = middle_images.reset_index(drop=True)

    # Define the grid size for plotting
    ncols = (len(false_positives_df) + nrows - 1) // nrows  # Adjust the calculation as needed
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle('False Positive Images')

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i, row in false_positives_df.iterrows():
        img_name = row[img_name_col]
        img_path = os.path.join(base_img_path, img_name.split('_')[1], img_name)  # Adjust path construction as needed
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'ID: {row["id"]} conf: {row[conf_col]:.2f} true: {row[true_label_col]} pred: {row[pred_label_col]}')
        axes[i].axis('off')

    # Hide any empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Adjust DPI for higher resolution images
    plt.show()

