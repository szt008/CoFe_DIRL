import torch
import torch.nn as nn
import hyper_param as hp
import torch.nn.functional as F

class CriticNet(nn.Module):
    def __init__(self, device, max_velocity=20):
        super(CriticNet, self).__init__()

        self.max_velocity = max_velocity

        self.state_dim = 21 * hp.feature_num * hp.history_frame_num
        self.action_dim = 4
        # Q1 architecture
        self.l1 = nn.Linear(self.state_dim + self.action_dim, 4096).to(device)
        self.l2 = nn.Linear(4096, 2048).to(device)
        self.l3 = nn.Linear(2048, 1).to(device)

        # Q2 architecture
        self.l4 = nn.Linear(self.state_dim + self.action_dim, 4096).to(device)
        self.l5 = nn.Linear(4096, 2048).to(device)
        self.l6 = nn.Linear(2048, 1).to(device)
		
    def forward(self, state, action):
        state = state.contiguous().view(-1, self.state_dim)
        
        action_lon = action[0].contiguous().view(-1, 1) / self.max_velocity
        action_lat = action[1].contiguous().view(-1, 3)

        sa = torch.cat([state, action_lon, action_lat], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        state = state.contiguous().view(-1, self.state_dim)
        action_lon = action[0].contiguous().view(-1, 1)
        action_lat = action[1].contiguous().view(-1, 3)

        sa = torch.cat([state, action_lon, action_lat], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1