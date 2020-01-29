import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPolicy_64x64(nn.Module):
    """policy network with a convolutional head best for 64x64 images"""

    def __init__(self, nb_actions):
        super(ConvPolicy_64x64, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=1, padding=2)  # 64x64x8
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)  # 32x32x16
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=1)  # 16x16x32
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 8x8x64
        self.conv5 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 4x4x64
        self.conv6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 2x2x64
        self.conv_layers = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6]
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, nb_actions)
        self.print_info()

    def print_info(self):
        print("Initializing conv64x64 policy network!")
        print(self)
        conv_params = sum([sum([p.numel() for p in l.parameters()]) for l in self.conv_layers])
        print("Convolutional Params: {}".format(conv_params))
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        for l in self.conv_layers:
            x = F.leaky_relu(l(x), 0.2)
        x = F.leaky_relu(self.fc1(x.view(-1, 256)))
        return self.fc2(x)


class ConvValue_64x64(nn.Module):
    """Value network with a convolutional head for 64x64 images"""

    def __init__(self):
        super(ConvValue_64x64, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=1, padding=2)  # 64x64x8
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)  # 32x32x16
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=1)  # 16x16x32
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 8x8x64
        self.conv5 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 4x4x64
        self.conv6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 2x2x64
        self.conv_layers = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6]
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.print_info()

    def print_info(self):
        print("Initializing conv64x64 value network!")
        print(self)
        conv_params = sum([sum([p.numel() for p in l.parameters()]) for l in self.conv_layers])
        print("Convolutional Params: {}".format(conv_params))
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        for l in self.conv_layers:
            x = F.leaky_relu(l(x), 0.2)
        x = F.leaky_relu(self.fc1(x.view(-1, 256)))
        return self.fc2(x)


class FCPolicy(nn.Module):
    """fully connected policy network"""

    def __init__(self, state_size, nb_actions):
        super(FCPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, nb_actions)
        self.print_info()

    def print_info(self):
        print("Initializing fully connected policy network!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2, self.fc3]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class FCValue(nn.Module):
    """fully connected value network"""

    def __init__(self, state_size):
        super(FCValue, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.print_info()

    def print_info(self):
        print("Initializing fully connected policy network!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2, self.fc3]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class FCQValueContinuous(nn.Module):
    """fully connected q value network"""

    def __init__(self, state_size, nb_actions):
        super(FCQValueContinuous, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fca = nn.Linear(nb_actions, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.print_info()

    def print_info(self):
        print("Initializing fully connected policy network!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2, self.fc3]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x, a):
        x = F.leaky_relu(self.fc1(x))
        a = F.leaky_relu(self.fca(a))
        x = torch.cat((x, a), dim=-1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
