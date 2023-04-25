import sys

import torch

import models
from utils import *
from args import parse_train_args
from datasets import make_dataset
from tqdm import tqdm

def loss_compute(args, model, criterion, outputs, targets):
    if args.loss == 'CrossEntropy':
        loss = criterion(outputs[0], targets)
    elif args.loss == 'MSE':
        loss = criterion(outputs[0], nn.functional.one_hot(targets, num_classes=10).type(torch.FloatTensor).to(args.device))

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        W = []
        B = []
        for fc_layer in model.fc:
            if isinstance(fc_layer, nn.Linear):
                W.append(fc_layer.weight)
                if fc_layer.bias is not None:
                    B.append(fc_layer.bias) 
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss += lamb * sum([torch.sum(w ** 2) for w in W])
        loss += lamb_feature * torch.sum(features ** 2)

    return loss

def trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]), logfile)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()
        outputs = model(inputs)
        
        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets)
        else:
            if args.loss == 'CrossEntropy':
                loss = criterion(outputs[0], targets)
            elif args.loss == 'MSE':
                loss = criterion(outputs[0], nn.functional.one_hot(targets, num_classes=10).type(torch.FloatTensor).to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)
    scheduler.step()



def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')

    print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
    print_and_save('--------------------- Training -------------------------------', logfile)
    for epoch_id in tqdm(range(args.epochs)):

        trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    logfile.close()


def main():
    args = parse_train_args()
    name = args.dataset + "-" + args.model \
           + "-" + args.loss + "-" + args.optimizer \
           + "-width_" + str(args.width) \
           + "-depth_relu_" + str(args.depth_relu) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-bias_" + str(args.bias) + "-" + "lr_" + str(args.lr) \
           + "-data_augmentation_" + str(args.data_augmentation) \
           + "-" + "epochs_" + str(args.epochs) + "-" + "seed_" + str(args.seed) + "-" + "wd_" + str(args.weight_decay) \


    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(manualSeed = args.seed)
    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=args.data_augmentation)
    
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth_relu = args.depth_relu, depth_linear = args.depth_linear, fc_bias=args.bias, num_classes=num_classes).to(device)
    else:
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, num_classes=num_classes, fc_bias=args.bias).to(device)
    

    train(args, model, trainloader)


if __name__ == "__main__":
    main()
