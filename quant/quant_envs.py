import torch
import torch.nn as nn
import copy

from .modules.gru import GRU as PYGRU
from .modules.ops import Mul, Add
from .qmodules.quantizers import INT_Quantizer, OP_INT_Quantizer
from .qmodules.quant_layers import INT_Conv2D, INT_Linear
from .qmodules.quant_ops import Quant_sigmoid, Quant_tanh, Quant_mult, Quant_add


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value


def create_quantizer(type, n_bits, all_positive):
    quantizer_types = ['INT_Quantizer']
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'INT_Quantizer' in type:
        quantizer = INT_Quantizer(n_bits, all_positive)
    else:
        raise NotImplementedError('Quantizer type {} is not implemented.'.format(type))
    return quantizer

def recur_rpls_layers(args, model, layer_type, rpls_layer_type, weight_quantizer, act_quantizer):
    """ Recursively replace layers of a given type with another type within a model.
    Args:
        model: the model to be searched.
        layer_type: the type of the layer to be replaced.
        rpls_layer_type: the type of the layer to be replaced with.
    Returns:
        A list of layers of the given type.
    """

    for name, module in model.named_children():
        weight_quantizer = create_quantizer(weight_quantizer.__class__.__name__, args.n_bits_w, all_positive=False)
        act_quantizer = create_quantizer(act_quantizer.__class__.__name__, args.n_bits_a, all_positive=False)
        if isinstance(module, layer_type):
            setattr(model, name, rpls_layer_type(module, weight_quantizer, act_quantizer))
        else:
            recur_rpls_layers(args, module, layer_type, rpls_layer_type, weight_quantizer, act_quantizer)

def create_op_quantizer(type, n_bits, all_positive):
    quantizer_types = ['OP_INT_Quantizer']
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'OP_INT_Quantizer' in type:
        quantizer = OP_INT_Quantizer(n_bits, all_positive)
    else:
        raise NotImplementedError('Quantizer type {} is not implemented.'.format(type))
    return quantizer

def recur_rpls_ops(args, model, op_type, rpls_op_type, *quantizers):
    """ Recursively replace layers of a given type with another type within a model.
    Args:
        model: the model to be searched.
        layer_type: the type of the layer to be replaced.
        rpls_layer_type: the type of the layer to be replaced with.
    Returns:
        A list of layers of the given type.
    """
    sigmoid_quantizer, tanh_quantizer, \
    mult_quantizer, add_quantizer = quantizers
    for name, module in model.named_children():        
        if isinstance(module, op_type):
            # print('Replace {} with {}'.format(op_type, rpls_op_type))
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_quantizer = create_op_quantizer(sigmoid_quantizer.__class__.__name__, args.n_bits_w, all_positive=False)
                setattr(model, name, rpls_op_type(sigmoid_quantizer))
            elif isinstance(module, torch.nn.Tanh):
                tanh_quantizer = create_op_quantizer(tanh_quantizer.__class__.__name__, args.n_bits_w, all_positive=False)
                setattr(model, name, rpls_op_type(tanh_quantizer))
            elif isinstance(module, Mul):
                mult_quantizer = create_op_quantizer(mult_quantizer.__class__.__name__, args.n_bits_w, all_positive=False)
                setattr(model, name, rpls_op_type(mult_quantizer))
            elif isinstance(module, Add):
                add_quantizer = create_op_quantizer(add_quantizer.__class__.__name__, args.n_bits_w, all_positive=False)
                setattr(model, name, rpls_op_type(add_quantizer))
            else:
                raise NotImplementedError('Operation type {} is not implemented.'.format(op_type))
            # print("model: ", model)
        else:
            recur_rpls_ops(args, module, op_type, rpls_op_type, *quantizers)


def recur_rpls_gru(model):
    """ Recursively replace GRU module with the self-defined pytorch GRU module.
    Args:
        model: the model to be searched.
    Returns:
        A list of layers of the given type.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.GRU):
            setattr(model, name, PYGRU(input_size = module.input_size,
                                       hidden_size = module.hidden_size,
                                       num_layers = module.num_layers,
                                       batch_first = module.batch_first,
                                       bias=module.bias is not None)
                    )
        else:
            recur_rpls_gru(module)

class Base_GRUQuantEnv(object):
    """ Base class for quantization environment
    Args:
        args: arguments
        model: the model to be quantized.
    """
    def __init__(self, model, args=AttrDict()):
        self.args = args
        self.model = model
        
        self.n_bits_w = args.n_bits_w
        self.n_bits_a = args.n_bits_a
        
        self.fq_layers_hash = {
            nn.Conv2d: INT_Conv2D,
            nn.Linear: INT_Linear,
        }
        self.fq_ops_hash = {
            nn.Sigmoid: Quant_sigmoid,
            nn.Tanh: Quant_tanh,
            Mul: Quant_mult,
            Add: Quant_add,
        }

        # quantizers
        self.weight_quantizer, self.act_quantizer, \
        self.sigmod_quantizer, self.tanh_quantizer, \
        self.mult_quantizer, self.add_quantizer  = self.set_quantizer()

        # float model
        self.pygru_model = self.create_pygru_model(copy.deepcopy(self.model))
        self.pygru_model = self.load_model(self.pygru_model)
        
        # quantized model
        self.q_model = self.create_quantized_model(copy.deepcopy(self.pygru_model))
        
    def load_model(self, model):
        pretrained_model = self.args.pretrained_model
        use_pretrained = bool(pretrained_model)
        
        if use_pretrained:
            model.load_state_dict(torch.load(pretrained_model))
            print("Load pretrained model from {}".format(pretrained_model))
        else:
            print("No pretrained model is loaded.")
        return model
    
    def set_quantizer(self):
        print('INT Quantizers are used.')
        weight_quantizer = INT_Quantizer(self.n_bits_w, all_positive=False)
        act_quantizer = INT_Quantizer(self.n_bits_a, all_positive=False)
   
        sigmod_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        tanh_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        mult_quantizer = OP_INT_Quantizer(self.n_bits_w, all_positive=False)
        add_quantizer = OP_INT_Quantizer(self.n_bits_w, all_positive=False)
        
        
        return weight_quantizer, act_quantizer, sigmod_quantizer, tanh_quantizer, mult_quantizer, add_quantizer

    def create_pygru_model(self, model):
        """ Create a pytorch GRU model from the original model.
        Args:
            model: the original model.
        Returns:
            A model with pytorch GRU module.
        """
        recur_rpls_gru(model)
        return model
           
    def create_quantized_model(self, model):
        """ Create a quantized model from the original model.
        Args:
            model: the original model.
        Returns:
            A quantized pygru model.
        """

        for op_type, rpls_op_type in self.fq_ops_hash.items():
            recur_rpls_ops(self.args, model, op_type, rpls_op_type, \
                self.sigmod_quantizer, self.tanh_quantizer, self.mult_quantizer, self.add_quantizer)
        
        for layer_type, rpls_layer_type in self.fq_layers_hash.items():
            recur_rpls_layers(self.args, model, layer_type, rpls_layer_type, self.weight_quantizer, self.act_quantizer)

        return model