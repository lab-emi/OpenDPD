import warnings
import pandas as pd
import torch


class PandasLogger:
    def __init__(self, path_save_file_best: str, path_log_file_hist: str, path_log_file_best: str, precision: int = 8):
        self.path_save_file_best = path_save_file_best
        self.path_log_file_hist = path_log_file_hist
        self.path_log_file_best = path_log_file_best
        self.list_log_headers = []
        self.list_log_rows = []
        self.precision = precision
        self.best_val_metric = None

    def add_row(self, list_header, list_value):
        self.list_log_headers = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.list_log_rows.append(row)

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
            print(f'>>> saving best model (%f -> %f {metric_name}) from epoch %d to %s' % (self.best_val_metric, best_criteria, epoch, self.path_save_file_best))
        if best_criteria < self.best_val_metric:
            best_epoch = epoch
            # Record the best epoch
            self.write_log_idx(best_epoch, self.path_log_file_best)
            torch.save(net.state_dict(), self.path_save_file_best)
            print(f'>>> saving best model (%f -> %f {metric_name}) from epoch %d to %s' % (self.best_val_metric, best_criteria, epoch, self.path_save_file_best))
            self.best_val_metric = best_criteria