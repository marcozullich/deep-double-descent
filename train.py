import torch
from torch import Tensor
import os


def accuracy(nn_output: Tensor, ground_truth: Tensor, k: int=1):
    '''
    Return accuracy@k for the given model output and ground truth
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in accuracy@k
    '''
    assert k <= nn_output.shape[1], f"k too big. Found: {k}. Max: {nn_output.shape[1]} inferred from the nn_output"
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler): # note: I've added a generic performance to replace accuracy
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad() 
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        y_hat = model(X)
        # 3. calculate the loss on the current mini-batch
        loss = loss_fn(y_hat, y)
        # 4. execute the backward pass given the current loss
        loss.backward()
        # 5. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 6. calculate the accuracy for this mini-batch
        acc = performance(y_hat, y)
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])

def train_model(model, dataloader, loss_fn, optimizer, num_epochs, checkpoint_loc=None, checkpoint_name="checkpoint.pt", performance=accuracy, lr_scheduler=None, device=None, lr_scheduler_step_on_epoch=True):

    # create the folder for the checkpoints (if it's not None)
    if checkpoint_loc is not None:
        os.makedirs(checkpoint_loc, exist_ok=True)

    if device is None:
        device = use_gpu_if_possible()
    
    model = model.to(device)
    model.train()

    # epoch loop
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        # added print for LR - activate if needed
        # print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        
        lr_scheduler_batch = lr_scheduler if not lr_scheduler_step_on_epoch else None

        train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler_batch)

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")

        # produce checkpoint dictionary -- but only if the name and folder of the checkpoint are not None
        if checkpoint_name is not None and checkpoint_loc is not None:
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_loc, checkpoint_name))
        
        if lr_scheduler is not None and lr_scheduler_step_on_epoch:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(loss_meter.avg)
            else:
                lr_scheduler.step()

    return loss_meter.sum, performance_meter.avg

def test_model(model, dataloader, performance=accuracy, loss_fn=None, device=None):
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()
    
    if device is None:
        device = use_gpu_if_possible()

    model = model.to(device)

    performance_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = model(X)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    return fin_loss, fin_perf

