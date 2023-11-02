__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "MIT License"
__version__ = "1.0"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

from steps import train_pa, train_dpd, run_dpd
from project import Project

if __name__ == '__main__':
    proj = Project()

    # PA Modeling
    if proj.step == 'train_pa':
        print("####################################################################################################")
        print("# Step: Train PA                                                                                   #")
        print("####################################################################################################")
        train_pa.main(proj)

    # DPD Learning
    elif proj.step == 'train_dpd':
        print("####################################################################################################")
        print("# Step: Train DPD                                                                                  #")
        print("####################################################################################################")
        train_dpd.main(proj)

    # Run DPD to Generate Predistorted PA Outputs
    elif proj.step == 'run_dpd':
        print("####################################################################################################")
        print("# Step: Run DPD                                                                                    #")
        print("####################################################################################################")
        run_dpd.main(proj)
    else:
        raise ValueError(f"The step '{proj.step}' is not supported.")