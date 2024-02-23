import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch CLIP Linear Probing')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='ViT-B/32', type=str, help='model architecture')
parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
parser.add_argument('--save-epoch', default=5, type=int, help='save the model every save_epoch')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.1, type=float, help='weight decay (default: 0.1)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: False)')
parser.add_argument("--warmup_length", type=int, default=500, help='initial warmup iterations for the scheduler (default: 500)')

parser.add_argument('--image-resolution', default=64, type=int, help='preprocessing image resolution (default: 64)')
parser.add_argument('--print-freq', '-p', default=75, type=int, help='print frequency (default: 75)')
parser.add_argument('--random-seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.set_defaults(augment=True)

# Grabbing cli arguments and printing results
args = parser.parse_args()
print_args = '*'*45
for key,value in args._get_kwargs():
    print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

print_args = print_args + '\n' + '*'*45
print(print_args)

directory = "/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)

save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(print_args, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from utils import ImageNet, PCAM, step_lr
import torchvision.transforms as transforms
from clip.ftmodel import ImageEncoder, ImageClassifier, LogisticRegressionHead
from utils.label_map import cifar10_labels, cifar10_classnames, cifar100_classnames, imagenet_classnames

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


def main():
    # Setting up data augmentation if needed 
    # TODO Check if we need data augmentations
    if args.augment:
        transform = transforms.Compose([
            transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_resolution),
            convert_image_to_rgb,
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([convert_image_to_rgb, transforms.ToTensor()])

    # Setting up testing dataset
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        classnames = cifar10_classnames
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        classnames = cifar100_classnames
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == 'ImageNet':
        classnames = imagenet_classnames
        train_loader = torch.utils.data.DataLoader(
            ImageNet(root='/nobackup-fast/ageng/datasets/ImageNet-1k/', train=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            ImageNet('/nobackup-fast/ageng/datasets/ImageNet-1k/', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # Loading zero-shot CLIP model
    image_encoder = ImageEncoder('ViT-B/32', keep_lang=True)
    classification_head = LogisticRegressionHead(input_size=512, output_size=len(classnames), normalize=True)
    delattr(image_encoder.model, 'transformer') #TODO perhaps we don't need to remove this param
    model = ImageClassifier(image_encoder, classification_head)
    preprocess = model.preprocess
    
    # Specifying gpu usage for model
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # Defining loss function and optimizer
    image_loss = torch.nn.CrossEntropyLoss().cuda()
    params = [p for p in model.module.classification_head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Defining learning rate scheduler
    scheduler = step_lr(optimizer, args.lr, args.epochs*len(train_loader))

    # Iterating through training epochs
    for epoch in range(0, args.epochs):

        # Train model for one epoch
        train(train_loader, model, preprocess, image_loss, optimizer, scheduler, epoch)

        # Evaluate model on testing dataset
        accuracy = validate(val_loader, model, preprocess, image_loss, epoch)

        # Save the checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model,
                'loss': accuracy,
            }, epoch + 1)


def train(train_loader, model, preprocess, image_loss, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to train mode
    model.train()

    end = time.time()
    num_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Update learning rate with scheduler
        scheduler(i + epoch*num_batches)

        # Redefining inputs to CLIP expected values
        inputs = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in inputs], dim=0).cuda()

        # Forward through CLIP vision encoder
        similarity = model(inputs)

        # Calculate the loss function for CLIP
        loss = image_loss(similarity, targets)
        
        # Backpropagate gradients and update model
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure top1 accuracy and record the loss
        accuracy = get_clip_accuracy(similarity.detach().clone(), targets)[0]
        batch_loss.update(loss.data, inputs.size(0))
        batch_accuracy.update(accuracy, inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Current Learning Rate: {lr}'.format(lr=get_lr(optimizer)))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))
    
    print('---------------> Training Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))


def validate(val_loader, model, preprocess, image_loss, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to evaluation mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Redefining inputs to CLIP expected values
        inputs = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in inputs], dim=0).cuda()

        # Forward through CLIP vision encoder
        with torch.no_grad():
            similarity = model(inputs)

        # Calculate the loss function for CLIP
        loss = image_loss(similarity, targets)

        # Measure top1 accuracy and record the loss
        accuracy = get_clip_accuracy(similarity.detach().clone(), targets)[0]
        batch_loss.update(loss.data, inputs.size(0))
        batch_accuracy.update(accuracy, inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(val_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_clip_accuracy(similarity, targets, topk=(1,)):
    similarity = similarity.softmax(dim=-1)
    batch_size = targets.size(0)
    maxk = max(topk)

    _, pred = similarity.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def save_checkpoint(state, epoch):
    directory = "/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = directory + 'checkpoint_{}.pth'.format(epoch)
    torch.save(state, filename)


class AverageMeter(object):
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


if __name__ == '__main__':
    main()