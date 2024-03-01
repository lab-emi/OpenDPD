import tkinter as tk
from tkinter import ttk
import os
import shutil
from tkinter import filedialog
from tkinter import messagebox
import subprocess
import shlex
import shutil
import glob
import os


class DPD_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DPD and Hyperparameters Settings")

        # Initialize all GUI components
        self.create_widgets()

    def create_widgets(self):
        # Create the tab control
        self.tab_control = ttk.Notebook(self.root)

        # Create three tabs
        self.tab0 = ttk.Frame(self.tab_control)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab0, text='DPD Experiment guidance')
        self.tab_control.add(self.tab1, text='DPD Settings')
        self.tab_control.add(self.tab2, text='Hyperparameters')
        self.images=[]
        # DPD Settings Tab
        Logo_image = tk.PhotoImage(file='/home/yizhuo/PycharmProjects/OpenDPD/pics/OpenDPDlogo.png')
        Logo_image = Logo_image.subsample(10, 10)
        # Keep a reference to the image so it's not garbage collected
        self.images.append(Logo_image)

        # Create a label with the image
        label = ttk.Label(self.tab0,image=Logo_image)
        label.grid(column=0, row=0,rowspan=2, pady=10, padx=10)

        # DPD Settings Tab
        dataset = tk.PhotoImage(file='/home/yizhuo/PycharmProjects/OpenDPD/pics/dataset.png')
        dataset = dataset.subsample(2, 2)
        # Keep a reference to the image so it's not garbage collected
        self.images.append(dataset)

        # Create a label with the image
        data_image = ttk.Label(self.tab0, image=dataset)
        data_image.grid(column=2, row=2, columnspan=2, rowspan=2, pady=10, padx=10)
        data_label = ttk.Label(self.tab0, text="Please upload dataset like follows:")
        data_label.grid(column=0, row=2, columnspan=2, pady=10, padx=10)

        # DPD Settings Tab
        spec = tk.PhotoImage(file='/home/yizhuo/PycharmProjects/OpenDPD/pics/spec.png')
        # Keep a reference to the image so it's not garbage collected
        self.images.append(spec)

        # Create a label with the image
        spec_image = ttk.Label(self.tab0, image=spec)
        spec_image.grid(column=2, row=4, rowspan=2,columnspan=2, pady=10, padx=10)
        spec_label = ttk.Label(self.tab0, text="Please upload spectrum settings file like follows:")
        spec_label.grid(column=0, row=4, columnspan=2, pady=10, padx=10)

        # Dataset Name setting
        self.Project_root_label = ttk.Label(self.tab0, text="Your project root:")
        self.Project_root_label.grid(column=1, row=0, pady=10, padx=10)

        # Entry widget for dataset_name
        self.Project_root_entry = tk.Entry(self.tab0, textvariable=tk.StringVar(value='/home/yizhuo/PycharmProjects/OpenDPD/'))
        self.Project_root_entry.grid(column=2, row=0, pady=10, padx=10)

        self.Download_label = ttk.Label(self.tab0, text="Your download root:")
        self.Download_label.grid(column=1, row=1, pady=10, padx=10)

        # Entry widget for dataset_name
        self.Download_label_entry = tk.Entry(self.tab0, textvariable=tk.StringVar(value='/home/yizhuo/PycharmProjects/OpenDPD/'))
        self.Download_label_entry.grid(column=2, row=1, pady=10, padx=10)

        # Dataset Name setting
        self.dataset_name_label = ttk.Label(self.tab1, text="1. Dataset Name:")
        self.dataset_name_label.grid(column=0,row=0,pady=10, padx=10)

        # Entry widget for dataset_name
        self.dataset_name_entry = ttk.Entry(self.tab1)
        self.dataset_name_entry.grid(column=1,row=0,pady=10,padx=10)


        self.load_data_label = ttk.Label(self.tab1, text="2. Load dataset")
        self.load_data_label.grid(column=0, row=1, pady=10,padx=10)

        self.Model_label = ttk.Label(self.tab1, text="3. Set Model Parameters")
        self.Model_label.grid(column=0, row=2, pady=10,padx=10)

        # Load spec.json Button
        self.load_spec_json_button = tk.Button(self.tab1, text="Load Spec.json", command=self.load_spec_json)
        self.load_spec_json_button.grid(column=1,row=1,pady=10, padx=10)

        # Load CSV Button
        ttk.Button(self.tab1, text="Load CSV", command=self.load_csv).grid(column=2,row=1,pady=10, padx=10)

        #steps
        steps = ['train_pa', 'train_dpd', 'run_dpd']
        k=0
        for step in steps:
            ttk.Button(self.tab1, text=f"Run {step}", command=lambda s=step: self.run_script(s)).grid(column=k,row=9, pady=10,padx=10)
            k+=1

        ttk.Button(self.tab1, text="Run shell script", command=self.run_shell_script).grid(column=0, row=10, pady=10, padx=10)
        ttk.Button(self.tab1, text="Download DPD data", command=self.download_data).grid(column=1, row=10, pady=10,padx=10)

        #  Hyperparameters Tab
        hyperparameters = {
            "accelerator":"Accelerator types.chose from mps, cpu,gpu",
            "devices":"Which accelerator to train on.",
            "frame_length":"Frame length of signals",
            "frame_stride":"stride_length length of signals",
            "seed":"Global random number seed.",
            "batch_size":"Batch size for training.",
            "n_epochs":"Number of epochs to train for.",
            "lr":"Learning rate"
        }
        m=0
        for param, explanation in hyperparameters.items():
            self.label = ttk.Label(self.tab2, text=f"{param}:")
            self.label.grid(column=0, row=m,padx=10,pady=10)
            self.label.bind("<Enter>", lambda event, widget=self.label, text=explanation: self.show_tooltip(event, widget, text))
            m+=1

        self.accelerator=tk.Entry(self.tab2,textvariable=tk.StringVar(value='cpu'))
        self.accelerator.grid(column=1, row=0,padx=10,pady=10)
        self.devices=tk.Entry(self.tab2,textvariable=tk.StringVar(value='0'))
        self.devices.grid(column=1, row=1,padx=10,pady=10)
        self.frame_length=tk.Entry(self.tab2,textvariable=tk.StringVar(value='50'))
        self.frame_length.grid(column=1, row=2,padx=10,pady=10)
        self.frame_stride=tk.Entry(self.tab2,textvariable=tk.StringVar(value='1'))
        self.frame_stride.grid(column=1, row=3,padx=10,pady=10)
        self.seed=tk.Entry(self.tab2,textvariable=tk.StringVar(value='0'))
        self.seed.grid(column=1, row=4,padx=10,pady=10)
        self.batch_size=tk.Entry(self.tab2,textvariable=tk.StringVar(value='64'))
        self.batch_size.grid(column=1, row=5,padx=10,pady=10)
        self.n_epochs=tk.Entry(self.tab2,textvariable=tk.StringVar(value='100'))
        self.n_epochs.grid(column=1, row=6,padx=10,pady=10)
        self.lr=tk.Entry(self.tab2,textvariable=tk.StringVar(value='1e-5'))
        self.lr.grid(column=1, row=7,padx=10,pady=10)


        #Model settings Tab
        Model = {
            "PA_backbone": "Modeling PA Recurrent layer type, chose from 'gmp', 'fcn', 'gru', 'dgru', 'lstm', 'vdlstm','ligru', 'pgjanet', 'dvrjanet','cnn1d', 'rvtdcnn', 'tcn'",
            "PA_hidden_size":"Hidden size of PA backbone",
            "PA_num_layers":"Number of layers of the PA backbone.",
            "DPD_backbone": "DPD model Recurrent layer type, chose from 'gmp', 'fcn', 'gru', 'dgru', 'lstm', 'vdlstm','ligru', 'pgjanet', 'dvrjanet','cnn1d', 'rvtdcnn', 'tcn'",
            "DPD_hidden_size": "Hidden size of DPD backbone.",
            "DPD_num_layers": "Number of layers of the DPD backbone."
        }


        n=3
        for param, explanation in Model.items():
            self.label = ttk.Label(self.tab1, text=f"{param}:")
            self.label.grid(column=0,row=n,pady=10, padx=10)
            self.label.bind("<Enter>", lambda event, widget=self.label, text=explanation: self.show_tooltip(event, widget, text))
            n+=1
        self.PA_backbone_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='dgru'))
        self.PA_backbone_entry.grid(column=1,row=3, pady=10, padx=10)
        self.PA_hidden_size_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='8'))
        self.PA_hidden_size_entry.grid(column=1,row=4, pady=10, padx=10)
        self.PA_num_layers_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='1'))
        self.PA_num_layers_entry.grid(column=1,row=5, pady=10, padx=10)
        self.DPD_backbone_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='dgru'))
        self.DPD_backbone_entry.grid(column=1,row=6, pady=10, padx=10)
        self.DPD_hidden_size_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='8'))
        self.DPD_hidden_size_entry.grid(column=1,row=7, pady=10, padx=10)
        self.DPD_num_layers_entry=tk.Entry(self.tab1,textvariable=tk.StringVar(value='1'))
        self.DPD_num_layers_entry.grid(column=1,row=8, pady=10, padx=10)
        # Add more widgets as needed...

        # Place the tab control in the window
        self.tab_control.pack(expand=1, fill="both")

    def load_csv(self):
        # Define the destination directory
        destination_dir = os.path.expanduser(self.Project_root_entry.get()+self.dataset_name_entry.get())
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Ask the user to select CSV files
        files = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        # Copy each selected file to the destination directory
        for file_path in files:
            base_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_dir, base_name)
            try:
                shutil.copy(file_path, destination_path)
                print(f"Copied {base_name} to {destination_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while copying {base_name}: {e}")

        messagebox.showinfo("Success", "All selected files have been copied successfully.")


    def run_script(self, step):
        # Build the command with arguments
        cmd = f"python main.py --step {step} --dataset_name {self.dataset_name_entry.get()}"
        # Add other arguments as needed here
        cmd = cmd + f" --PA_backbone {self.PA_backbone_entry.get()}"
        cmd = cmd + f" --PA_hidden_size {int(self.PA_hidden_size_entry.get())}"
        cmd = cmd + f" --PA_num_layers {int(self.PA_num_layers_entry.get())}"
        cmd = cmd + f" --DPD_backbone {self.DPD_backbone_entry.get()}"
        cmd = cmd + f" --DPD_hidden_size {int(self.DPD_hidden_size_entry.get())}"
        cmd = cmd + f" --DPD_num_layers {int(self.DPD_num_layers_entry.get())}"
        cmd = cmd + f" --accelerator {self.accelerator.get()}"
        cmd = cmd + f" --devices {int(self.devices.get())}"
        cmd = cmd + f" --frame_length {int(self.frame_length.get())}"
        cmd = cmd + f" --frame_stride {int(self.frame_stride.get())}"
        cmd = cmd + f" --seed {int(self.seed.get())}"
        cmd = cmd + f" --batch_size {int(self.batch_size.get())}"
        cmd = cmd + f" --n_epochs {int(self.n_epochs.get())}"
        cmd = cmd + f" --lr {float(self.lr.get())}"


        # Run the command
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("Script ran successfully.")
            print(stdout.decode())
        else:
            print("There was an error running the script.")
            print(stderr.decode())

    def load_spec_json(self):
        # Implement the logic to load spec.json file here
        destination_dir = os.path.expanduser(self.Project_root_entry.get() +'/dataset/'+ self.dataset_name_entry.get())
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Ask the user to select CSV files
        files = filedialog.askopenfilenames(
            title="Select spec.json File",
            filetypes=(("json files", "*.json"), ("All files", "*.*"))
        )

        # Copy each selected file to the destination directory
        for file_path in files:
            base_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_dir, base_name)
            try:
                shutil.copy(file_path, destination_path)
                print(f"Copied {base_name} to {destination_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while copying {base_name}: {e}")

        messagebox.showinfo("Success", "All selected files have been copied successfully.")
        print("Spec.json loading not implemented")

    def show_tooltip(self, event, widget, text):
        # Show tooltip with explanation
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.geometry(f"+{event.x_root + 20}+{event.y_root + 10}")
        label = tk.Label(tooltip, text=text, background="light yellow", borderwidth=1)
        label.pack()

        def on_leave(event):
            tooltip.destroy()

        widget.bind("<Leave>", on_leave)

    def run_shell_script(self):
        # Build the command with arguments
        cmd = "ssh train_all_pa.sh"

        # Run the command
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("Script ran successfully.")
            print(stdout.decode())
        else:
            print("There was an error running the script.")
            print(stderr.decode())

    def download_data(self):
        source_dir = self.Project_root_entry.get()+'/dpd_out'
        destination_dir = self.Download_label.get()

        # Make sure the destination directory exists, create if it doesn't
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Find all .csv files in the source directory
        csv_files = glob.glob(os.path.join(source_dir, '*.csv'))

        # Copy each .csv file to the destination directory
        for file_path in csv_files:
            # Get the base name of the file (without the directory part)
            file_name = os.path.basename(file_path)
            # Define the destination file path
            destination_file_path = os.path.join(destination_dir, file_name)
            # Copy the file
            shutil.copy(file_path, destination_file_path)
            print(f"Copied: {file_path} to {destination_file_path}")


def GUI():
    root = tk.Tk()
    app = DPD_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    GUI()