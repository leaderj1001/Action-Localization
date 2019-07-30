import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRNN(nn.Module):
    def __init__(self, temporal_size, bidirectional, dropout):
        super(HierarchicalRNN, self).__init__()
        self.hidden_size = [256, 1024]
        self.num_layer = 1
        self.bidirectional = bidirectional
        self.dropout = dropout

        if self.bidirectional:
            self.left_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=bidirectional, dropout=dropout)
            self.right_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=bidirectional, dropout=dropout)
            self.torso = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[1], batch_first=True, bidirectional=bidirectional, dropout=dropout)
            self.left_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=bidirectional, dropout=dropout)
            self.right_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=bidirectional, dropout=dropout)

            self.lstm = nn.LSTM(input_size=self.hidden_size[1] * 4, hidden_size=self.hidden_size[1] * 2, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            self.left_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, dropout=dropout)
            self.right_arm = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, dropout=dropout)
            self.torso = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[1], batch_first=True, dropout=dropout)
            self.left_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, dropout=dropout)
            self.right_leg = nn.LSTM(input_size=temporal_size, hidden_size=self.hidden_size[0], batch_first=True, dropout=dropout)

            self.lstm = nn.LSTM(input_size=self.hidden_size[1] * 2, hidden_size=self.hidden_size[1] * 2, batch_first=True, dropout=dropout)

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
    def __init__(self, joint_sequence_size, bidirectional, dropout):
        super(TraversalSequence, self).__init__()
        self.hidden_size = 2048
        self.num_layer = 1

        if bidirectional:
            self.lstm = nn.LSTM(input_size=joint_sequence_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_size=joint_sequence_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, hidden = self.lstm(x)
        return out


class Model(nn.Module):
    def __init__(self, temporal_size, joint_sequence_size=70, num_classes=80, dropout=0.0, mode='normal'):
        super(Model, self).__init__()
        self.hidden_size = 2048
        self.dense_hidden_size = 1024
        self.hierarchical_joints = 12
        self.mode = mode
        print('temporal_size :: ', temporal_size, 'joint_sequence_size :: ', joint_sequence_size, 'num classes :: ', num_classes)

        if mode == 'normal':
            self.hierarchical_lstm = HierarchicalRNN(temporal_size=temporal_size, bidirectional=False, dropout=dropout)
            self.traversal_lstm = TraversalSequence(joint_sequence_size=joint_sequence_size, bidirectional=False, dropout=dropout)

            self.hierarchical_dense = nn.Linear(self.hidden_size * self.hierarchical_joints, num_classes)
            self.traversal_dense = nn.Linear(self.hidden_size * temporal_size, num_classes)
        elif mode == 'dense':
            self.hierarchical_lstm = HierarchicalRNN(temporal_size=temporal_size, bidirectional=False, dropout=dropout)
            self.traversal_lstm = TraversalSequence(joint_sequence_size=joint_sequence_size, bidirectional=False, dropout=dropout)

            self.hierarchical_dense = nn.Sequential(
                nn.Linear(self.hidden_size * self.hierarchical_joints, self.dense_hidden_size),
                nn.BatchNorm1d(self.dense_hidden_size),
                nn.Dropout(dropout),
                nn.ReLU()
            )
            self.hierarchical_dense2 = nn.Linear(self.dense_hidden_size, num_classes)

            self.traversal_dense = nn.Sequential(
                nn.Linear(self.hidden_size * temporal_size, self.dense_hidden_size),
                nn.BatchNorm1d(self.dense_hidden_size),
                nn.Dropout(dropout),
                nn.ReLU()
            )

            self.traversal_dense2 = nn.Linear(self.dense_hidden_size, num_classes)
        elif mode == 'bilstm':
            self.hierarchical_lstm = HierarchicalRNN(temporal_size=temporal_size, bidirectional=True, dropout=dropout)
            self.traversal_lstm = TraversalSequence(joint_sequence_size=joint_sequence_size, bidirectional=True, dropout=dropout)

            self.hierarchical_dense = nn.Linear(self.hidden_size * 2 * self.hierarchical_joints, num_classes)
            self.traversal_dense = nn.Linear(self.hidden_size * 2 * temporal_size, num_classes)

    def forward(self, x1, x2, x3, x4, x5, traversal_x):
        batch = x1.size(0)

        hierarchical_out = self.hierarchical_lstm(x1, x2, x3, x4, x5)
        traversal_out = self.traversal_lstm(traversal_x)

        hierarchical_out = hierarchical_out.contiguous().view(batch, -1)
        traversal_out = traversal_out.contiguous().view(batch, -1)

        if self.mode == 'normal':
            hierarchical_out = self.hierarchical_dense(hierarchical_out)
            traversal_out = self.traversal_dense(traversal_out)
        elif self.mode == 'dense':
            hierarchical_out = self.hierarchical_dense(hierarchical_out)
            traversal_out = self.traversal_dense(traversal_out)

            hierarchical_out = self.hierarchical_dense2(hierarchical_out)
            traversal_out = self.traversal_dense2(traversal_out)
        elif self.mode == 'bilstm':
            hierarchical_out = self.hierarchical_dense(hierarchical_out)
            traversal_out = self.traversal_dense(traversal_out)

        out = hierarchical_out * 0.9 + traversal_out * 0.1
        return out
