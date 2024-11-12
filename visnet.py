import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import re

from pgl.utils import op
from pgl.message import Message

from visnet_utils import \
(
    AtomEncoder,
    FgEncoder,
    BondEncoder,
    CosineCutoff,
    rbf_class_mapping,
    Sphere,
    VecLayerNorm
)
import visnet_output_modules


class ViS_Graph(pgl.Graph):
    
    def recv(self, reduce_func, msg, recv_mode="dst"):
        """Recv message and aggregate the message by reduce_func.

        The UDF reduce_func function should has the following format.

        .. code-block:: python

            def reduce_func(msg):
                '''
                    Args:

                        msg: An instance of Message class.

                    Return:

                        It should return a tensor with shape (batch_size, out_dims).
                '''
                pass

        Args:

            msg: A dictionary of tensor created by send function..

            reduce_func: A callable UDF reduce function.

        Return:

            A tensor with shape (num_nodes, out_dims). The output for nodes with 
            no message will be zeros.

        """

        if not self._is_tensor:
            raise ValueError("You must call Graph.tensor()")

        if not isinstance(msg, dict):
            raise TypeError(
                "The input of msg should be a dict, but receives a %s" %
                (type(msg)))

        if not callable(reduce_func):
            raise TypeError("reduce_func should be callable")

        src, dst, eid = self.sorted_edges(sort_by=recv_mode)
        msg = op.RowReader(msg, eid)
        uniq_ind, segment_ids = self.get_segment_ids(
            src, dst, segment_by=recv_mode)
        bucketed_msg = Message(msg, segment_ids)
        output = reduce_func(bucketed_msg)
        x, vec = output
        
        x_output_dim = x.shape[-1]
        vec_output_dim1 = vec.shape[-1]
        vec_output_dim2 = vec.shape[-2]
        x_init_output = paddle.zeros(
            shape=[self._num_nodes, x_output_dim], dtype=x.dtype)
        x_final_output = paddle.scatter(x_init_output, uniq_ind, x)
        
        vec_init_output = paddle.zeros(
            shape=[self._num_nodes, vec_output_dim2, vec_output_dim1], dtype=vec.dtype)
        vec_final_output = paddle.scatter(vec_init_output, uniq_ind, vec)

        return x_final_output, vec_final_output


class ViSNetBlock(nn.Layer):
    
    def __init__(
        self,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        cutoff=5.0,
        max_num_neighbors=32,
    ):
        super().__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.atom_embedding = AtomEncoder(hidden_channels)
        self.fg_embedding = FgEncoder(hidden_channels)
        self.bond_embedding = BondEncoder(hidden_channels)
        
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        
        # self.bond_fuse = nn.Linear(hidden_channels + num_rbf, hidden_channels)
        self.bond_fuse = nn.Linear(num_rbf, hidden_channels)

        self.vis_mp_layers = nn.LayerList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            activation=activation,
            attn_activation=attn_activation,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm
        )
        vis_mp_class = ViS_MP
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        nn.initializer.XavierUniform(self.bond_fuse.weight)
        # self.bond_fuse.bias.data.fill_(0)
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        # self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(self, graph: pgl.Graph):

        # x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        graph.node_feat['atom_pos'] = paddle.to_tensor(graph.node_feat['atom_pos']).reshape((graph.num_nodes, -1))
        pos, edge_index = graph.node_feat['atom_pos'], graph.edges.T
       
        
        node_feat, edge_feat = graph.node_feat, graph.edge_feat

        # Embedding Layers
        x = self.atom_embedding(node_feat)
        fg = self.fg_embedding(node_feat)
        edge_attr = self.bond_embedding(edge_feat)

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        mask = edge_index[0] != edge_index[1]        
        edge_weight = paddle.zeros(edge_vec.shape[0])
        edge_vec = paddle.to_tensor(edge_vec)  # 确保 edge_vec 是一个 Paddle 张量
        edge_weight[mask] = paddle.norm(edge_vec[mask], axis=-1)
        
        dist_edge_attr = self.distance_expansion(edge_weight)
        edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        vec = paddle.zeros((x.shape[0], ((self.lmax + 1) ** 2) - 1, x.shape[1]))
        # TODO for test (no edge_attr). change back later
        # edge_attr = self.bond_fuse(paddle.concat([edge_attr, dist_edge_attr], axis=-1))
        edge_attr = self.bond_fuse(dist_edge_attr)
        
        vis_graph = ViS_Graph(
            num_nodes=x.shape[0],
            edges=edge_index.T,
            node_feat={'x': x, 'vec': vec},
            edge_feat={'r_ij': edge_weight, 'f_ij': edge_attr, 'd_ij': edge_vec},
        )
        vis_graph._graph_node_index = graph._graph_node_index
        vis_graph._graph_edge_index = graph._graph_edge_index
        
        # ViS-MP Layers
        for attn in self.vis_mp_layers:
            dx, dvec = attn(vis_graph)
            x = x + dx
            vec = vec + dvec
            vis_graph.node_feat['x'] = x
            vis_graph.node_feat['vec'] = vec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec, fg


class ViS_MP(nn.Layer):
    
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super().__init__()
        
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        
        self.act = nn.Silu()
        self.attn_activation = nn.Silu()
        
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias_attr=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias_attr=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias_attr=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
    
    def reset_parameters(self):
        # self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.initializer.XavierUniform(self.q_proj.weight)
        nn.initializer.XavierUniform(self.k_proj.weight)
        nn.initializer.XavierUniform(self.v_proj.weight)
        nn.initializer.XavierUniform(self.o_proj.weight)
        nn.initializer.XavierUniform(self.s_proj.weight)

        nn.initializer.XavierUniform(self.vec_proj.weight)
        nn.initializer.XavierUniform(self.dk_proj.weight)
        nn.initializer.XavierUniform(self.dv_proj.weight)
        
    def forward(self, graph: pgl.Graph):
        """
        graph:
            - num_nodes,
            - edges <--> edge_index ,
            - node_feat <--> x, vec,
            - edge_feat <--> r_ij, f_ij, d_ij,
        """
        
        x, vec, r_ij, f_ij, d_ij = \
        (
            graph.node_feat["x"], 
            graph.node_feat["vec"], 
            graph.edge_feat["r_ij"],
            graph.edge_feat["f_ij"],
            graph.edge_feat["d_ij"]
        )
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape([-1, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape([-1, self.num_heads, self.head_dim])
        v = self.v_proj(x).reshape([-1, self.num_heads, self.head_dim])
        dk = self.act(self.dk_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim])
        dv = self.act(self.dv_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim])

        vec1, vec2, vec3 = paddle.split(self.vec_proj(vec), 3, axis=-1)
        vec_dot = (vec1 * vec2).sum(axis=1)
        
        def _send_func(src_feat, dst_feat, edge_feat):
            
            q_i = dst_feat["q"]
            k_j, v_j, vec_j = \
            (
                src_feat["k"], 
                src_feat["v"],
                src_feat["vec"]
            )
            dk, dv, r_ij, d_ij = \
            (
                edge_feat["dk"],
                edge_feat["dv"],
                edge_feat["r_ij"],
                edge_feat["d_ij"]
            )
            
            attn = (q_i * k_j * dk).sum(axis=-1)
            attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

            v_j = v_j * dv
            v_j = (v_j * attn.unsqueeze(2)).reshape([-1, self.hidden_channels])

            s1, s2 = paddle.split(self.act(self.s_proj(v_j)), 2, axis=1)
            vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

            return {
                'x': v_j,
                'vec': vec_j
            }
            
        def _recv_func(msg: pgl.Message):
            
            x, vec = msg["x"], msg["vec"]
            return msg.reduce(x, pool_type="sum"), msg.reduce(vec, pool_type="sum")
        
        msg = graph.send(
            message_func=_send_func,
            node_feat={
                "q": q,
                "k": k,
                "v": v,
                "vec": vec
            },
            edge_feat={
                "dk": dk,
                "dv": dv,
                "r_ij": r_ij,
                "d_ij": d_ij
            }
        )

        x, vec_out = graph.recv(
            reduce_func=_recv_func,
            msg=msg
        )
        
        o1, o2, o3 = paddle.split(self.o_proj(x), 3, axis=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        
        return dx, dvec
    
    
def create_visnet_model(args, mean=None, std=None):
    visnet_args = dict(
        lmax=args["lmax"],
        vecnorm_type=args["vecnorm_type"],
        trainable_vecnorm=args["trainable_vecnorm"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        hidden_channels=args["embedding_dimension"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        attn_activation=args["attn_activation"],
        cutoff=args["cutoff"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "ViSNetBlock":
        representation_model = ViSNetBlock(**visnet_args)
    else:
        raise ValueError(f"Unknown model {args['model']}.")

    # create output network
    output_prefix = "Equivariant"
    output_model = getattr(visnet_output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args['out_dimension'], args["activation"]
    )

    model = ViSNet(
        representation_model,
        output_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
    )

    return model


def load_visnet_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = paddle.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            print(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_visnet_model(args)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    return model.to(device)


class ViSNet(nn.Layer):
    def __init__(
            self,
            representation_model,
            output_model,
            reduce_op="sum",
            mean=None,
            std=None,
    ):
        super().__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.reduce_op = reduce_op

        mean = paddle.to_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = paddle.to_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(self, graph: pgl.Graph):
        x, v, fg = self.representation_model(graph)
        x = self.attention(x, fg)
        x = self.output_model.pre_reduce(x, v)
        x = pgl.math.segment_pool(x, graph.graph_node_id, pool_type=self.reduce_op)
        return x
    
    def attention(self, x, fg,):
        """
        Perform attention between x and fg after pooling fg.
        
        Args:
            x (Tensor): Input features (batch_size, hidden_channels)
            fg (Tensor): Functional group features (batch_size, fg_channels)
        
        Returns:
            Tensor: The updated features after attention.
        """
        #import logging
        #logging.debug(print(paddle.transpose(fg, perm=[1, 0]).shape))
        # Calculate attention scores (dot-product attention as an example)
        attention_scores = paddle.matmul(x,  paddle.transpose(fg, perm=[1, 0]))  # Shape: [batch_size, hidden_channels]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, axis=-1)  # Shape: [batch_size, hidden_channels]
        
        # Weighted sum of fg features (no need for bmm since both are 2D)
        fg_weighted = paddle.matmul(attention_weights, fg)  # Shape: [batch_size, hidden_channels]
        
        # Combine x and the attended fg
        attended_x = x + fg_weighted  # The final attended features

        return attended_x
    '''
    def attention(self, x, fg,):
        """
        Perform attention between x and fg after pooling fg.
        
        Args:
            x (Tensor): Input features (batch_size, hidden_channels)
            fg (Tensor): Functional group features (batch_size, fg_channels)
        
        Returns:
            Tensor: The updated features after attention.
        """
        #import logging
        #logging.debug(print(paddle.transpose(fg, perm=[1, 0]).shape))
        # Calculate attention scores (dot-product attention as an example)
        self.WQ = nn.Linear(80,80)
        self.WK = nn.Linear(80,80)
        q = self.WQ(x)
        k = self.WK(fg)

        attention_scores = paddle.matmul(q,  paddle.transpose(k, perm=[1, 0]))  # Shape: [batch_size, hidden_channels]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, axis=-1)  # Shape: [batch_size, hidden_channels]
        
        # Weighted sum of fg features (no need for bmm since both are 2D)
        fg_weighted = paddle.matmul(attention_weights, fg)  # Shape: [batch_size, hidden_channels]
        
        # Combine x and the attended fg
        #attended_x = x + fg_weighted  # The final attended features

        return fg_weighted 
    '''