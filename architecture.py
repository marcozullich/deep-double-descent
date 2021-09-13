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