import torch
import torch.nn as nn
import hyper_param as hp
from agent.gaussian_dropout import GaussianDropout

class ActorNet(nn.Module):
    def __init__(self, device, max_velocity=20):
        super(ActorNet, self).__init__()
        self.input_feature_num = 21 * hp.feature_num * hp.history_frame_num
        self.max_velocity = max_velocity

        self.hidden_layer_1 = nn.Linear(self.input_feature_num, 4096).to(device)
        # self.activation_f_1 = nn.Tanh().to(device)
        self.activation_f_1 = nn.Softsign().to(device)
        # self.activation_f_1 = nn.Sigmoid().to(device)
        self.dropout_1 = GaussianDropout(p=0.25)

        self.hidden_layer_2 = nn.Linear(4096, 2048).to(device)
        self.activation_f_2 = nn.Softsign().to(device)
        # self.activation_f_2 = nn.ReLU().to(device)
        self.dropout_2 = GaussianDropout(p=0.25)
       
        self.hidden_layer_3_lon = nn.Linear(2048, 1024).to(device)
        self.hidden_layer_4_lat = nn.Linear(2048, 1024).to(device)
        
        self.lon_out_layer = nn.Linear(1024, 1).to(device)
        self.lat_out_layer = nn.Linear(1024, 3).to(device)

        self.activation_f_lon_out = nn.Softsign().to(device)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, execute_dropout=False):
        x = x.contiguous().view(-1, self.input_feature_num)
        # printColor(x)
        x = self.hidden_layer_1(x)
        x = self.activation_f_1(x)
        # printColor(x)
        x = self.dropout_1(x, execute_dropout)
        x = self.hidden_layer_2(x)
        x = self.activation_f_2(x)
        x = self.dropout_2(x, execute_dropout)
        # lon_output = torch.tanh(self.lon_out_layer(x)) * 6 - 2 # -8 < ax < 4 m^2/s

        self.raw_lon_out = self.lon_out_layer(self.hidden_layer_3_lon(x))
        self.raw_lat_out = self.lat_out_layer(self.hidden_layer_4_lat(x))

        # printColor(self.raw_lon_out)
        lon_output = torch.tanh(self.raw_lon_out) * self.max_velocity / 2 + self.max_velocity / 2 # -0 < vx < 20 m/s
        # lon_output = self.activation_f_lon_out(self.raw_lon_out) * 10 + 10 # -0 < vx < 20 m/s
        
        lat_output = self.sm(self.raw_lat_out)
        return (lon_output, lat_output)