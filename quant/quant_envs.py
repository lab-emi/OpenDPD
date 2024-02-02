import torch
import torch.nn as nn
import copy

from .modules.gru import GRU as PYGRU
from .modules.ops import Mul, Add, Sqrt, Pow
from .qmodules.quantizers import (
    Identity_Quantizer, INT_Quantizer, OP_INT_Quantizer
    )
from .qmodules.quant_layers import INT_Conv2D, INT_Linear, INT_Pass
from .qmodules.quant_ops import Quant_sigmoid, Quant_tanh, Quant_mult, Quant_add, Quant_sqrt, Quant_pow


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value


def create_quantizer(type, n_bits, all_positive, act_or_weight):
    quantizer_types = ['INT_Quantizer', 'Identity_Quantizer',
                       'Drf_Act_Quantizer', 'Drf_Weight_Quantizer',
                       'IAO_Quantizer', 
                       'FP8_Quantizer', 
                       'PACT_Quantizer',                       
                    ]
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'INT_Quantizer' in type:
        quantizer = INT_Quantizer(n_bits, all_positive)
    elif 'Identity_Quantizer' in type:
        quantizer = Identity_Quantizer(n_bits, all_positive)
    else:
        raise NotImplementedError('Quantizer type {} is not implemented.'.format(type))
    return quantizer

def recur_rpls_layers(args, model, layer_type=nn.Conv2d,
                      rpls_layer_type=INT_Conv2D,
                      weight_quantizer=INT_Quantizer(8, all_positive=False),
                      act_quantizer=INT_Quantizer(8, all_positive=False)):
    """ Recursively replace layers of a given type with another type within a model.
    Args:
        model: the model to be searched.
        layer_type: the type of the layer to be replaced.
        rpls_layer_type: the type of the layer to be replaced with.
    Returns:
        A list of layers of the given type.
    """

    for name, module in model.named_children():
        weight_quantizer = create_quantizer(weight_quantizer.__class__.__name__, args.n_bits_w, all_positive=False, act_or_weight='weight')
        act_quantizer = create_quantizer(act_quantizer.__class__.__name__, args.n_bits_a, all_positive=False, act_or_weight='act')
        if isinstance(module, layer_type):
            print('Replace {} with {}'.format(layer_type, rpls_layer_type))
            setattr(model, name, rpls_layer_type(module, weight_quantizer, act_quantizer))
        else:
            recur_rpls_layers(args, module, layer_type, rpls_layer_type, weight_quantizer, act_quantizer)

def create_op_quantizer(type, n_bits, all_positive):
    quantizer_types = ['OP_INT_Quantizer', 'Identity_Quantizer', 'Drf_Act_Quantizer', 'IAO_Quantizer', 'FP8_Quantizer']
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'OP_INT_Quantizer' in type:
        quantizer = OP_INT_Quantizer(n_bits, all_positive)
    elif 'Identity_Quantizer' in type:
        quantizer = Identity_Quantizer(n_bits, all_positive)
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
    sigmoid_quantizer, tanh_quantizer, mult_quantizer, add_quantizer, \
    sqrt_quantizer, pow_quantizer = quantizers
    
    for name, module in model.named_children():        
        if isinstance(module, op_type):
            # print('Replace {} with {}'.format(op_type, rpls_op_type))
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_quantizer = create_op_quantizer(sigmoid_quantizer.__class__.__name__, sigmoid_quantizer.bits, sigmoid_quantizer.all_positive)
                setattr(model, name, rpls_op_type(sigmoid_quantizer))
            elif isinstance(module, torch.nn.Tanh):
                tanh_quantizer = create_op_quantizer(tanh_quantizer.__class__.__name__, tanh_quantizer.bits, tanh_quantizer.all_positive)
                setattr(model, name, rpls_op_type(tanh_quantizer))
            elif isinstance(module, Mul):
                mult_quantizer = create_op_quantizer(mult_quantizer.__class__.__name__, mult_quantizer.bits, mult_quantizer.all_positive)
                setattr(model, name, rpls_op_type(mult_quantizer))
            elif isinstance(module, Add):
                add_quantizer = create_op_quantizer(add_quantizer.__class__.__name__, add_quantizer.bits, add_quantizer.all_positive)
                setattr(model, name, rpls_op_type(add_quantizer))
            elif isinstance(module, Sqrt):
                sqrt_quantizer = create_op_quantizer(sqrt_quantizer.__class__.__name__, sqrt_quantizer.bits, sqrt_quantizer.all_positive)
                setattr(model, name, rpls_op_type(sqrt_quantizer))
            elif isinstance(module, Pow):
                pow_quantizer = create_op_quantizer(pow_quantizer.__class__.__name__, pow_quantizer.bits, pow_quantizer.all_positive)
                setattr(model, name, rpls_op_type(module, pow_quantizer))
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
            Sqrt: Quant_sqrt,
            Pow: Quant_pow,
        }

        self.last_layer_type = INT_Pass

        # quantizers
        self.weight_quantizer, self.act_quantizer,  \
        self.sigmod_quantizer, self.tanh_quantizer, \
        self.mult_quantizer, self.add_quantizer,    \
        self.sqrt_quantizer, self.pow_quantizer        = self.set_quantizer()

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
        # weight_quantizer = IAO_Quantizer(bits=self.n_bits_w, all_positive=False, act_or_weight='weight')
        # act_quantizer = IAO_Quantizer(bits=self.n_bits_a, all_positive=False, act_or_weight='act')
        # weight_quantizer = Drf_Weight_Quantizer(bits=self.n_bits_w, all_positive=False)
        # act_quantizer = Drf_Act_Quantizer(bits=self.n_bits_a, all_positive=True)
        
        # weight_quantizer = FP8_Quantizer(self.n_bits_w, all_positive=False)
        # act_quantizer = Identity_Quantizer(self.n_bits_a, all_positive=False)
   
        sigmod_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        tanh_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        mult_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        add_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        

        # sigmod_quantizer = Drf_Act_Quantizer(self.n_bits_a, all_positive=False)
        # tanh_quantizer = Drf_Act_Quantizer(self.n_bits_a, all_positive=False)
        # mult_quantizer = Drf_Act_Quantizer(self.n_bits_w, all_positive=False)
        # add_quantizer = Drf_Act_Quantizer(self.n_bits_w, all_positive=False)
        # sqrt_quantizer = OP_INT_Quantizer(bits=16, all_positive=False)
        # pow_quantizer = OP_INT_Quantizer(bits=16, all_positive=False)   
        sqrt_quantizer = Identity_Quantizer(self.n_bits_w, all_positive=False)
        pow_quantizer = Identity_Quantizer(self.n_bits_w, all_positive=False)
        
        
        return weight_quantizer, act_quantizer, sigmod_quantizer, tanh_quantizer, mult_quantizer, add_quantizer, \
               sqrt_quantizer, pow_quantizer

    def create_pygru_model(self, model):
        """ Create a pytorch GRU model from the original model.
        Args:
            model: the original model.
        Returns:
            A model with pytorch GRU module.
        """
        def _reset_parameters(model, hidden_size):
            for name, param in model.named_parameters():
                num_gates = int(param.shape[0] / hidden_size)
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                if 'weight' in name:
                    for i in range(0, num_gates):
                        nn.init.orthogonal_(param[i * hidden_size:(i + 1) * hidden_size, :])
                if 'x2h.weight' in name:
                    for i in range(0, num_gates):
                        nn.init.xavier_uniform_(param[i * hidden_size:(i + 1) * hidden_size, :])

        def _reset_pygru(model):
            for name, module in model.named_children():
                if isinstance(module, PYGRU):
                    print("::: Reset pytorch GRU module.")
                    _reset_parameters(module, module.hidden_size)
                else:
                    _reset_pygru(module)
        
        # replace GRU module with pytorch GRU module
        recur_rpls_gru(model)
        
        # reset parameters
        _reset_pygru(model)
 
        return model

    def unquantize_last_layer(self, model, last_layer_name='fc_out'):
        """ Unquantize the last layer of the model.
        Args:
            model: the model to be added with the last layer.
            last_layer_name: the name of the last layer.
        Returns:
            A model with the a unquantized last layer.
        """
        for name, module in model.named_children():
            if name == last_layer_name:
                print("Unquantize the last layer: ", name)
                module.weight_quantizer = Identity_Quantizer()
                module.act_quantizer = Identity_Quantizer()
            else:
                self.unquantize_last_layer(module, last_layer_name)
    
    def set_first_layer(self, model, first_layer_name='x2h'):
        """ Set the first layer attributes of the model.
        """
        for name, module in model.named_children():
            if name == first_layer_name:
                print("Set the first layer: ", name)
                module.act_quantizer.bits = 16
            else:
                self.set_first_layer(module, first_layer_name)
    
    def set_last_layer_quant(self, model, last_layer_name='fc_out'):
        """ Set the last layer attributes of the model.
        """
        for name, module in model.named_children():
            if name == last_layer_name:
                print("quant the output")
                module.out_quant = True
            else:
                self.set_last_layer_quant(module, last_layer_name)
                
    def create_quantized_model(self, model):
        """ Create a quantized model from the original model.
        Args:
            model: the original model.
        Returns:
            A quantized pygru model.
        """

        for op_type, rpls_op_type in self.fq_ops_hash.items():
            recur_rpls_ops(self.args, model, op_type, rpls_op_type, \
                self.sigmod_quantizer, self.tanh_quantizer, self.mult_quantizer, self.add_quantizer, \
                self.sqrt_quantizer, self.pow_quantizer)
        
        for layer_type, rpls_layer_type in self.fq_layers_hash.items():
            recur_rpls_layers(self.args, model, layer_type, rpls_layer_type, self.weight_quantizer, self.act_quantizer)
        
        # self.set_first_layer(model)
        # self.unquantize_last_layer(model)
        self.set_last_layer_quant(model)
        
        return model