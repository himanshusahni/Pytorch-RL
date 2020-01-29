import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPolicy_64x64(nn.Module):
    """policy network with a convolutional head best for 64x64 images"""

    def __init__(self, state_shape, action_space):
        super(ConvPolicy_64x64, self).__init__()
        nb_actions = action_space.n
        channels = state_shape[0]
        self.conv1 = nn.Conv2d(channels, 8, 5, stride=1, padding=2)  # 64x64x8
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

    def __init__(self, state_shape):
        super(ConvValue_64x64, self).__init__()
        channels = state_shape[0]
        self.conv1 = nn.Conv2d(channels, 8, 5, stride=1, padding=2)  # 64x64x8
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


class ConvTrunk_64x64(nn.Module):
    """Value network with a convolutional head for 64x64 images"""

    def __init__(self, state_shape):
        super(ConvTrunk_64x64, self).__init__()
        channels = state_shape[0]
        self.conv1 = nn.Conv2d(channels, 8, 5, stride=1, padding=2)  # 64x64x8
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)  # 32x32x16
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 16x16x32
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 8x8x64
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 4x4x64
        self.conv6 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 2x2x64
        self.conv_layers = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6]
        self.output_size = 2 * 2 * 64
        self.print_info()

    def print_info(self):
        print("Initializing conv64x64 trunk network!")
        print(self)
        conv_params = sum([sum([p.numel() for p in l.parameters()]) for l in self.conv_layers])
        print("Convolutional Params: {}".format(conv_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        for l in self.conv_layers:
            x = F.leaky_relu(l(x), 0.2)
        return x.flatten(start_dim=1)


class FCTrunk(nn.Module):
    """trunk of a fully connected AC network"""

    def __init__(self, observation_space):
        super().__init__()
        state_size = observation_space.shape[0]
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_size = 256

    def print_info(self):
        print("Initializing fully connected trunk!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc1, self.fc2]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return F.leaky_relu(self.fc2(x))


class FCPolicyHead(nn.Module):
    """policy head for a fully connected AC network"""

    def __init__(self, input_size, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, action_shape)

    def print_info(self):
        print("Initializing fully connected policy head!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc,]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc2(x)


class FCValueHead(nn.Module):
    """value head for a fully connected AC network"""

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def print_info(self):
        print("Initializing fully value head!")
        print(self)
        fc_params = sum([sum([p.numel() for p in l.parameters()]) for l in [self.fc,]])
        print("FC params: {}".format(fc_params))
        print("Total trainable params: {}".format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc2(x)


class ActorCritic(nn.Module):
    """
    Combined fully connected AC network with a common
    trunk for policy and value
    """
    def __init__(self, trunk_arch, observation_space, action_space):
        super().__init__()
        self.trunk = trunk_arch(observation_space)
        self.policy_head = FCPolicyHead(self.trunk.output_size, action_space)
        self.value_head = FCValueHead(self.trunk.output_size)

    def print_info(self):
        self.trunk.print_info()
        self.policy_head.print_info()
        self.value_head.print_info()

    def parameters(self, recurse=True):
        return list(self.trunk.parameters()) + \
               list(self.policy_head.parameters()) + \
               list(self.value_head.parameters())

    def V_parameters(self, recurse=True):
        return list(self.trunk.parameters()) + \
               list(self.value_head.parameters())

    def pi_parameters(self, recurse=True):
        return list(self.trunk.parameters()) + \
               list(self.policy_head.parameters())

    def forward(self, x):
        x = self.trunk(x)
        return self.policy_head(x), self.value_head(x)

    def V(self, x):
        """only return values"""
        x = self.trunk(x)
        return self.value_head(x)

    def pi(self, x):
        """only return policy logits"""
        x = self.trunk(x)
        return self.policy_head(x)

    def to(self, device):
        self.trunk.to(device)
        self.policy_head.to(device)
        self.value_head.to(device)




class FCPolicy(nn.Module):
    """fully connected policy network"""

    def __init__(self, observation_space, action_space):
        super(FCPolicy, self).__init__()
        state_size = observation_space[0]
        nb_actions = action_space.n
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

    def __init__(self, observation_space):
        super(FCValue, self).__init__()
        state_size = observation_space[0]
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
