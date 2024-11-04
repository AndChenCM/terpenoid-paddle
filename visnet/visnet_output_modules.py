from abc import ABCMeta, abstractmethod

import paddle
import paddle.nn as nn
import numpy as np

__all__ = ["Scalar"]


class GatedEquivariantBlock(nn.Layer):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
            self,
            hidden_channels,
            out_channels,
            intermediate_channels=None,
            activation="silu",
            scalar_activation=False,
    ):
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias_attr=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias_attr=False)

        act_class = nn.Silu()
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class,
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class if scalar_activation else None

    def reset_parameters(self):
        nn.initializer.XavierUniform(self.vec1_proj.weight)
        nn.initializer.XavierUniform(self.vec2_proj.weight)
        nn.initializer.XavierUniform(self.update_net[0].weight)
        nn.initializer.XavierUniform(self.update_net[2].weight)

    def forward(self, x, v):
        vec1 = paddle.norm(self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)

        x = paddle.concat([x, vec1], axis=-1)
        x, v = paddle.split(self.update_net(x), 2, axis=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class OutputModel(nn.Layer, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super().__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v):
        return

    def post_reduce(self, x):
        return x



class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, out_channels, activation="silu", allow_prior_model=True):
        super().__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.LayerList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                activation=activation,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                out_channels,
                activation=activation,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0

class Pretrain_Output(OutputModel):
    def __init__(self, hidden_channels, out_channels, activation="silu", allow_prior_model=True):
        super().__init__(allow_prior_model=allow_prior_model)
      
        self.output_network = nn.LayerList([
            GatedEquivariantBlock(
                hidden_channels  ,
                hidden_channels // 2,
                activation=activation,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                out_channels,
                activation=activation,
                scalar_activation=False,
            ),
        ])
        
       
        #self.Blr_loss = nn.MSELoss()
        self.Blr_loss = nn.SmoothL1Loss()
        self.Bar_loss = nn.SmoothL1Loss()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def _get_Blr_loss(self, feed_dict, node_repr):

        loss = self.Blr_loss(node_repr, paddle.to_tensor(feed_dict['Bl_bond_length']))
        
        return loss
    
    def _get_Bar_loss(self, feed_dict, node_repr):

        
        loss = self.Bar_loss(node_repr, paddle.to_tensor(feed_dict['Ba_bond_angle'] / np.pi))
        return loss
    
    def pre_reduce(self, x_orig,v_orig,feed_dict):

        
        x_orig_node_i_repr = paddle.gather(x_orig, paddle.to_tensor(feed_dict['Ba_node_i']))
        x_orig_node_j_repr = paddle.gather(x_orig, paddle.to_tensor(feed_dict['Ba_node_j']))
        x_orig_node_k_repr = paddle.gather(x_orig, paddle.to_tensor(feed_dict['Ba_node_k']))
        x_orig_node_ijk_repr = paddle.concat([x_orig_node_i_repr, x_orig_node_j_repr, x_orig_node_k_repr], 1)
        v_orig_node_i_repr = paddle.gather(v_orig, paddle.to_tensor(feed_dict['Ba_node_i']))
        v_orig_node_j_repr = paddle.gather(v_orig, paddle.to_tensor(feed_dict['Ba_node_j']))
        v_orig_node_k_repr = paddle.gather(v_orig, paddle.to_tensor(feed_dict['Ba_node_k']))
        v_orig_node_ijk_repr = paddle.concat([v_orig_node_i_repr, v_orig_node_j_repr, v_orig_node_k_repr], 2)
        
        for layer in self.output_network:
            x_orig_node_ijk_repr, v_orig_node_ijk_repr = layer(x_orig_node_ijk_repr, v_orig_node_ijk_repr)
        
        loss = self._get_Bar_loss(feed_dict, x_orig_node_ijk_repr)

        return loss  
    

