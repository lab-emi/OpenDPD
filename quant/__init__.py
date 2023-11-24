from .quant_envs import Base_GRUQuantEnv, AttrDict
from .modules import Mul, Add, Sqrt, Pow


def get_quant_model(proj, net_dpd):
    if proj.args.quant:
        quant_env_args = AttrDict({
            'n_bits_w': proj.args.n_bits_w,
            'n_bits_a': proj.args.n_bits_a,
            'pretrained_model': proj.args.pretrained_model,
        })
        quant_env = Base_GRUQuantEnv(net_dpd, quant_env_args)
        if proj.args.q_pretrain:
            net_dpd = quant_env.pygru_model
            print("::: Pretraining DPD Model: ", net_dpd)
        else:
            net_dpd = quant_env.q_model
            print("::: Quantizing DPD Model: ", net_dpd)
            
    return net_dpd