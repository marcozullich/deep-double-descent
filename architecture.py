from torch import nn

class Mlp(nn.Module):
    def get_module(self, in_size, width, bias, activation, batch_norm):
        module = []
        if batch_norm:
            module += [nn.BatchNorm1d(in_size)]
        module += [nn.Linear(in_size, width, bias=bias), activation()]
        return nn.Sequential(*module)
        

    def __init__(self, width, depth=2, in_size=28*28, out_size=10, bias=True, activ=nn.ReLU, activ_out=nn.Identity, batch_norm=False):
        assert depth >= 2, f"Need depth to be at least 2. Found {depth}"
        
        super().__init__()
        module_list = [nn.Flatten(), self.get_module(in_size, width, bias, activ, batch_norm=False)]
        module_list += [self.get_module(width, width, bias, activ, batch_norm)] * (depth - 2)
        module_list += [self.get_module(width, out_size, bias, activation=activ_out, batch_norm=batch_norm)]

        self.module = nn.Sequential(*module_list)
    
    def forward(self, x):
        return self.module(x)

class CNN(nn.Module):
    def get_module(self, in_channels, width, bias, activation, batch_norm, max_pool=True, padding=0):
        module = []
        if batch_norm:
            module += [nn.BatchNorm1d(in_channels)]
        module += [nn.Conv2d(in_channels, width, kernel_size=3, padding=padding, bias=bias), activation()]
        if max_pool:
            module.append(nn.MaxPool2d(2))
        return nn.Sequential(*module)
    
    def __init__(self, width, num_modules, in_channels=3, activation=nn.ReLU, batch_norm=False, conv_padding=0, max_pool_each=2, num_classes=10):
        super().__init__()
        print(width, num_modules, in_channels)
        assert num_modules > 1, f"Expected num_modules>1, got {num_modules}"

        module = [self.get_module(in_channels, width, bias=True, activation=activation, batch_norm=batch_norm, padding=conv_padding, max_pool=(1 % max_pool_each == 0))]

        module += [self.get_module(width, width, bias=True, activation=activation, batch_norm
        =batch_norm, padding=conv_padding, max_pool=((index + 2) % max_pool_each == 0)) for index in range(num_modules - 1)]

        self.conv = nn.Sequential(*module)

        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(width, num_classes)
        )
    
    def forward(self, x):
        return self.classification(self.conv(x))

        