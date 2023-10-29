import warnings
import pandas as pd
import torch
from utils import util




class PandasLogger:
    def __init__(self, logfile, precision=8):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []
        self.precision = precision

    def add_row(self, list_header, list_value):
        self.list_header = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.loglist.append(row)

    def write_csv(self, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    # def write_log(self, log_stat):
    #     # Create Log List
    #     list_log_headers = []
    #     list_log_values = []
    #     for k, v in log_stat.items():
    #         list_log_headers.append(k)
    #         list_log_values.append(v)
    #
    #     # Write Log
    #     self.add_row(list_log_headers, list_log_values)
    #     self.write_csv()

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

    # def write_log_idx(self, idx, logfile=None):
    #     if len(self.list_header) == 0:
    #         warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
    #     else:
    #         loglist_best = [self.loglist[idx]]
    #         df = pd.DataFrame(loglist_best, columns=self.list_header)
    #         if logfile is not None:
    #             df.to_csv(logfile, index=False)
    #         else:
    #             df.to_csv(self.logfile, index=False)

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)

            # Format the float columns based on the specified precision
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = df[col].apply(lambda x: round(x, self.precision))

            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def save_best_model(self, best_val, net, save_file, logger, logfile_best, epoch, val_stat):
        best_criteria = val_stat['NMSE']
        if epoch == 0:
            best_val = best_criteria
            torch.save(net.state_dict(), save_file)
            best_epoch = epoch
            logger.write_log_idx(best_epoch, logfile_best)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
        if best_criteria <= best_val:
            best_val = best_criteria
            best_epoch = epoch
            # Record the best epoch
            logger.write_log_idx(best_epoch, logfile_best)
            torch.save(net.state_dict(), save_file)
            print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
        return best_val