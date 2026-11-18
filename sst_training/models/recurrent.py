"""Original DSPG ConvLSTM equations, with clear state and shape handling."""

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        self.hidden_dim = options["hidden_dim"]
        self.conv = nn.Conv2d(options["input_dim"] + self.hidden_dim, 4 * self.hidden_dim, 3, padding=1)

    def forward(self, values: torch.Tensor, state: tuple) -> tuple:
        hidden, cell = state
        gates = self.conv(torch.cat((values, hidden), dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
        return output_gate.sigmoid() * cell.tanh(), cell


class ConvLSTM(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        self.hidden_dim = options["hidden_dim"]
        self.cell_list = nn.ModuleList([ConvLSTMCell(options)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = (values.shape[0], self.hidden_dim) + tuple(values.shape[-2:])
        hidden, cell = values.new_zeros(shape), values.new_zeros(shape)
        outputs = []
        for step in range(values.shape[1]):
            hidden, cell = self.cell_list[0](values[:, step], (hidden, cell))
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class ConvBiLSTM(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        self.forward_lstm = ConvLSTM(options)
        self.backward_lstm = ConvLSTM(options)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        forward = self.forward_lstm(values)
        backward = self.backward_lstm(values.flip(1)).flip(1)
        return torch.cat((forward, backward), dim=2)
