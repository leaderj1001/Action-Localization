import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRNN(nn.Module):
    def __init__(self, temporal_size):
        super(HierarchicalRNN, self).__init__()
        self.hidden_size = [128, 512]

        self.left_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True)
        self.right_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True)
        self.torso = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[1], batch_first=True)
        self.left_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True)
        self.right_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True)

        self.lstm = nn.LSTM(input_size=self.hidden_size[1] * 2, hidden_size=self.hidden_size[1] * 2, batch_first=True)

    def forward(self, x1, x2, x3, x4, x5):
        left_arm, left_arm_hidden = self.left_arm(x1)
        right_arm, right_arm_hidden = self.right_arm(x2)
        torso, torso_hidden = self.torso(x3)
        left_leg, left_leg_hidden = self.left_leg(x4)
        right_leg, right_leg_hidden = self.right_leg(x5)

        out = torch.cat((left_arm, right_arm, torso, left_leg, right_leg), dim=2)
        out, hidden = self.lstm(out)
        return out


class TraversalSequence(nn.Module):
    def __init__(self, joint_sequence_size):
        super(TraversalSequence, self).__init__()
        self.hidden_size = 512

        self.lstm = nn.LSTM(input_size=joint_sequence_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x):
        out, hidden = self.lstm(x)
        return out


class Model(nn.Module):
    def __init__(self, temporal_size, joint_sequence_size=28, num_classes=80):
        super(Model, self).__init__()
        self.hidden_size = 512
        self.hierarchical_joints = 6

        self.hierarchical_lstm = HierarchicalRNN(temporal_size=temporal_size)
        self.traversal_lstm = TraversalSequence(joint_sequence_size=joint_sequence_size)

        self.hierarchical_dense = nn.Linear(self.hidden_size * 2 * self.hierarchical_joints, num_classes)
        self.traversal_dense = nn.Linear(self.hidden_size * temporal_size, num_classes)

    def forward(self, x1, x2, x3, x4, x5, traversal_x):
        batch = x1.size(0)

        hierarchical_out = self.hierarchical_lstm(x1, x2, x3, x4, x5).contiguous().view(batch, -1)
        traversal_out = self.traversal_lstm(traversal_x).contiguous().view(batch, -1)

        hierarchical_out = F.softmax(self.hierarchical_dense(hierarchical_out), dim=-1)
        traversal_out = F.softmax(self.traversal_dense(traversal_out), dim=-1)

        out = hierarchical_out * 0.9 + traversal_out
        return out
