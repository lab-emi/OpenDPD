__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import pandas as pd

# Load the files
input_data = pd.read_csv('../datasets/DPA_100MHz/Input.csv')
output_data = pd.read_csv('../datasets/DPA_100MHz/Output.csv')

def partition_data(input_df, output_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Partition the input and output datasets into train, validation, and test sets based on the given ratios.

    Args:
    - input_df (pd.DataFrame): Input datasets.
    - output_df (pd.DataFrame): Output datasets.
    - train_ratio (float): Ratio for training set.
    - val_ratio (float): Ratio for validation set.
    - test_ratio (float): Ratio for test set.

    Returns:
    - tuple of DataFrames: (train_input, train_output, val_input, val_output, test_input, test_output)
    """

    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios should sum to 1"

    # Compute the number of samples for each set
    total_samples = len(input_df)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the datasets
    train_input = input_df.iloc[:train_end]
    train_output = output_df.iloc[:train_end]

    val_input = input_df.iloc[train_end:val_end]
    val_output = output_df.iloc[train_end:val_end]

    test_input = input_df.iloc[val_end:]
    test_output = output_df.iloc[val_end:]

    return train_input, train_output, val_input, val_output, test_input, test_output


# Split datasets with default ratio of 60-20-20
train_input, train_output, val_input, val_output, test_input, test_output = partition_data(input_data, output_data)

# Save the partitions into separate files
export_path = './'
train_input.to_csv(export_path + "train_input.csv", index=False)
train_output.to_csv(export_path + "train_output.csv", index=False)
val_input.to_csv(export_path + "val_input.csv", index=False)
val_output.to_csv(export_path + "val_output.csv", index=False)
test_input.to_csv(export_path + "test_input.csv", index=False)
test_output.to_csv(export_path + "test_output.csv", index=False)
