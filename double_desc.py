import torch
import mnist, kmnist
import train
import architecture
import argparse
import asyncio
import pickle

def get_width_schedule(parallel_train, max_width, jump=25):
    # widths = torch.Tensor([2**i for i in range(7)])
    # seq_size = ((max_width - 100) // jump) + 1
    # if seq_size > 0:
    #     widths = torch.cat([widths, torch.linspace(100, (max_width//jump) * jump, seq_size)])
    #     if max_width % jump > 0:
    #         widths =  torch.cat([widths, max_width])
    widths = torch.linspace(1, 512, 125)
    return widths.split(parallel_train)
    

def network_train(width, depth, device, trainloader, testloader, dict_store, optimizer=torch.optim.Adam, num_epochs=5, net="mlp"):
    net_func = {
        "mlp": architecture.Mlp(width, depth, batch_norm=False),
        "cnn": architecture.CNN(width, depth, 1)
        }
    net = net_func[net]
    optim = optimizer(net.parameters())
    train_loss, _ = train.train_model(net, trainloader, torch.nn.CrossEntropyLoss(), optim, num_epochs, device=device)
    test_loss, _ = train.test_model(net, testloader, loss_fn=torch.nn.CrossEntropyLoss(), device=device)
    dict_store[width] = {"train":train_loss, "test":test_loss}


async def async_group_train(widths, depth, device, trainloader, testloader, result_dict, optimizer=torch.optim.Adam, num_epochs=5, net="mlp"):
    async for width in AsyncIterator(widths):
        width = int(width.item())
        print(f"--Started training with width {width}")
        network_train(width, depth, device, trainloader, testloader, result_dict, optimizer, num_epochs, net=net)

class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration

# async def run_group_train(widths, depth, device, trainloader, testloader, result_dict, optimizer=torch.optim.Adam, num_epochs=5):
#     await asyncio.gather(async_group_train(widths, depth, device, trainloader, testloader, result_dict, optimizer=torch.optim.Adam, num_epochs=5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hidden", type=int, default=3)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--parallel_train", type=int, default=1, help="Number of nets that will train in parallel. Defaults to 1, implying no parallelism.")
    parser.add_argument("--max_width", type=int, default=1000)
    parser.add_argument("--jump", type=int, default=25)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--label_noise_pct", type=float, default=0.0)
    parser.add_argument("--net", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--dataset", choices=["mnist", "kmnist"], default="mnist")
    
    args = parser.parse_args()

    assert not (args.use_gpu and args.use_cpu), "Specify NO MORE THAN one of use_cpu and use_gpu"
    if args.use_gpu:
        device = "cuda:0"
    if args.use_cpu:
        device = "cpu"
    if (not args.use_gpu) and (not args.use_cpu):
        device = train.use_gpu_if_possible()
    
    assert args.max_width >= 64, f"--max_width must be >= 64. Found {args.max_width}"

    assert args.label_noise_pct >= 0.0 and args.label_noise_pct <= 1.0, f"--label_noise_pct must be between 0 and 1. Found {args.label_noise_pct}"

    dataset_fn = {"mnist": mnist.get_MNIST, "kmnist":kmnist.get}
    trainloader, testloader, _, _ = dataset_fn[args.dataset](batch_size_train=args.batch_size, batch_size_test=args.batch_size, num_workers=8, label_noise_pct=args.label_noise_pct)

    widths_schedule = get_width_schedule(args.parallel_train, args.max_width)
    print("schedule\n", widths_schedule)

    result_dict = {}
    for group_id, async_widths in enumerate(widths_schedule):
        print(f"----STARTING GROUP {group_id} OF {len(widths_schedule)}")
        asyncio.run(async_group_train(async_widths, args.num_hidden+1, device, trainloader, testloader, result_dict=result_dict, num_epochs=args.num_epochs, net=args.net))
    
    #pickle.dump(open("results.pkl", "wb"), result_dict)
    torch.save(result_dict, "results_cnn_kmnist_tr2.pt")
