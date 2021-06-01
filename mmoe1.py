import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer


class MMOELayer(nn.Module):

    def __init__(self, input_dim, num_tasks, num_experts, output_dim, device, seed=6666):
        super(MMOELayer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
        torch.nn.init.normal_(w)
        self.expert_kernel = nn.Parameter(w).to(device)
        self.gate_kernels = []
        for i in range(self.num_tasks):
            w = torch.Tensor(torch.rand(input_dim, self.num_experts))
            torch.nn.init.normal_(w)
            self.gate_kernels.append(nn.Parameter(w).to(device))

    def forward(self, inputs):
        outputs = []
        expert_out = torch.mm(inputs, self.expert_kernel)
        expert_out = torch.reshape(expert_out, (-1, self.output_dim, self.num_experts))
        for i in range(self.num_tasks):
            gate_out = torch.mm(inputs, self.gate_kernels[i])
            gate_out = torch.nn.functional.softmax(gate_out, dim=1)
            gate_out = torch.unsqueeze(gate_out, dim=1).repeat(1, self.output_dim, 1)
            output = torch.sum(torch.mul(expert_out, gate_out), dim=2)
            outputs.append(output)
        return outputs


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, task_dnn_units=None, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu'):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.tasks = tasks
        self.task_dnn_units = task_dnn_units
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.mmoe_layer = MMOELayer(dnn_hidden_units[-1], num_tasks, num_experts, expert_dim, device)
        if task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, dnn_hidden_units) for _ in range(num_tasks)])
        self.tower_network = nn.ModuleList([nn.Linear(expert_dim, 1, bias=False) for _ in range(num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        self.to(device)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        mmoe_outs = self.mmoe_layer(dnn_output)
        if self.task_dnn_units is not None:
            mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]

        task_outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            logit = self.tower_network[i](mmoe_out)
            output = self.out[i](logit)
            task_outputs.append(output)

        task_outputs = torch.cat(task_outputs, -1)
        return task_outputs
