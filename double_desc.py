import torch
import mnist
import train
import architecture
import argparse
import asyncio
import pickle

def get_width_schedule(parallel_train, max_width, jump=25):
    widths = torch.Tensor([2**i for i in range(7)])
    seq_size = ((max_width - 100) // jump) + 1
    if seq_size > 0:
        widths = torch.cat([widths, torch.linspace(100, (max_width//jump) * jump, seq_size)])
        if max_width % jump > 0:
            widths =  torch.cat([widths, max_width])
    return widths.split(parallel_train)
    

def network_train(width, depth, device, trainloader, testloader, dict_store, optimizer=torch.optim.Adam, num_epochs=5):
    net = architecture.Mlp(width, depth)
    optim = optimizer(net.parameters())
    train_loss, _ = train.train_model(net, trainloader, torch.nn.CrossEntropyLoss(), optim, num_epochs, device=device)
    test_loss, _ = train.test_model(net, testloader, loss_fn=torch.nn.CrossEntropyLoss(), device=device)
    dict_store[width] = {"train":train_loss, "test":test_loss}


async def async_group_train(widths, depth, device, trainloader, testloader, result_dict, optimizer=torch.optim.Adam, num_epochs=5):
    async for width in AsyncIterator(widths):
        width = int(width.item())
        print(f"--Started training with width {width}")
        network_train(width, depth, device, trainloader, testloader, result_dict, optimizer, num_epochs)

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
    
    args = parser.parse_args()

    assert not (args.use_gpu and args.use_cpu), "Specify AT LEAST one of use_cpu and use_gpu"
    if args.use_gpu:
        device = "cuda:0"
    if args.use_cpu:
        device = "cpu"
    if (not args.use_gpu) and (not args.use_cpu):
        device = train.use_gpu_if_possible()
    
    assert args.max_width >= 64, f"--max_width must be >= 64. Found {args.max_width}"

    trainloader, testloader, _, _ = mnist.get_MNIST(batch_size_train=args.batch_size, batch_size_test=args.batch_size, num_workers=8)

    widths_schedule = get_width_schedule(args.parallel_train, args.max_width)
    print(widths_schedule)

    result_dict = {}
    for group_id, async_widths in enumerate(widths_schedule):
        print(f"----STARTING GROUP {group_id} OF {len(widths_schedule)}")
        asyncio.run(async_group_train(async_widths, args.num_hidden+1, device, trainloader, testloader, result_dict=result_dict, num_epochs=args.num_epochs))
    
    #pickle.dump(open("results.pkl", "wb"), result_dict)
    torch.save(result_dict, "results.pt")
