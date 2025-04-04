__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import warnings
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.columns import Columns


class PandasLogger:
    def __init__(self, path_save_file_best: str, path_log_file_hist: str, path_log_file_best: str, precision: int = 8):
        self.path_save_file_best = path_save_file_best
        self.path_log_file_hist = path_log_file_hist
        self.path_log_file_best = path_log_file_best
        self.list_log_headers = []
        self.list_log_rows = []
        self.precision = precision
        self.best_val_metric = None
        self.console = Console()

    def add_row(self, list_header, list_value):
        self.list_log_headers = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.list_log_rows.append(row)
        
        # Create a rich table instead of just printing the row
        self._display_stats_table(row)

    def _display_stats_table(self, stats):
        # Create two tables - one for general info and one for metrics
        table_general = Table(title="General Information")
        table_metrics = Table(title="Training Metrics")
        
        # Add columns for both tables
        table_general.add_column("Metric", style="cyan", no_wrap=True)
        table_general.add_column("Value", style="green")
        
        table_metrics.add_column("Metric", style="cyan", no_wrap=True)
        table_metrics.add_column("Value", style="green")
        
        # Group metrics by type
        general_metrics = []
        train_metrics = []
        val_metrics = []
        test_metrics = []
        
        # First pass - categorize metrics
        for key in stats.keys():
            if key == "TRAIN_LOSS" or key.startswith("TRAIN_"):
                train_metrics.append(key)
            elif key.startswith("VAL_"):
                val_metrics.append(key)
            elif key.startswith("TEST_"):
                test_metrics.append(key)
            else:
                general_metrics.append(key)
        
        # Sort each category
        general_metrics.sort()
        train_metrics.sort()
        val_metrics.sort()
        test_metrics.sort()
        
        # Process general metrics (left table)
        for key in general_metrics:
            value = stats[key]
            
            # Format floating point values
            if isinstance(value, float):
                # For general metrics, we may want to show different precision based on the value
                if key == "LR" or abs(value) < 0.01:
                    # More precision for small values like learning rate
                    formatted_value = f"{value:.8f}"
                else:
                    formatted_value = f"{value:.{self.precision}f}"
            else:
                formatted_value = str(value)
                
            # Determine row style
            if key == 'EPOCH':
                table_general.add_row(key, formatted_value, style="bold")
            else:
                table_general.add_row(key, formatted_value)
        
        # Format function for right table - ensures consistent decimal places
        def format_metric_value(value):
            if isinstance(value, float):
                # Ensure exactly self.precision decimal places
                return f"{value:.{self.precision}f}"
            else:
                return str(value)
        
        # Process training metrics (right table)
        for key in train_metrics:
            value = stats[key]
            formatted_value = format_metric_value(value)
            table_metrics.add_row(key, formatted_value, style="magenta")
        
        # Process validation metrics (right table)
        for key in val_metrics:
            value = stats[key]
            formatted_value = format_metric_value(value)
            table_metrics.add_row(key, formatted_value, style="blue")
        
        # Process test metrics (right table)
        for key in test_metrics:
            value = stats[key]
            formatted_value = format_metric_value(value)
            table_metrics.add_row(key, formatted_value, style="red")
        
        # Display tables side by side
        self.console.print(Columns([table_general, table_metrics]))

    def write_csv(self, logfile=None):
        if len(self.list_log_headers) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.list_log_rows, columns=self.list_log_headers)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.path_log_file_hist, index=False)

    def write_log(self, log_stat):
        # Create Log List
        list_log_headers = []
        list_log_values = []
        for k, v in log_stat.items():
            list_log_headers.append(k)

            # Check if the value is a floating point number
            if isinstance(v, float):
                # Format the floating point number based on the specified precision
                format_str = "{:." + str(self.precision) + "f}"
                v = format_str.format(v)

            list_log_values.append(v)

        # Write Log
        self.add_row(list_log_headers, list_log_values)
        self.write_csv()

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_log_headers) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.list_log_rows[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_log_headers)

            # Format the float columns based on the specified precision
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = df[col].apply(lambda x: round(x, self.precision))

            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.path_log_file_hist, index=False)

    def save_best_model(self, net, epoch, val_stat, metric_name='ACPR_AVG'):
        best_criteria = val_stat[metric_name]
        if epoch == 0:
            self.best_val_metric = best_criteria
            torch.save(net.state_dict(), self.path_save_file_best)
            best_epoch = epoch
            self.write_log_idx(best_epoch, self.path_log_file_best)
            self.console.print(f'[bold green]>>> saving best model ({self.best_val_metric} -> {best_criteria} {metric_name}) from epoch {epoch} to {self.path_save_file_best}[/bold green]')
        if best_criteria < self.best_val_metric:
            best_epoch = epoch
            # Record the best epoch
            self.write_log_idx(best_epoch, self.path_log_file_best)
            torch.save(net.state_dict(), self.path_save_file_best)
            self.console.print(f'[bold green]>>> saving best model ({self.best_val_metric} -> {best_criteria} {metric_name}) from epoch {epoch} to {self.path_save_file_best}[/bold green]')
            self.best_val_metric = best_criteria