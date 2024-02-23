import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch CLIP Reprogrammer')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="ImageNet", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='ViT-B/32', type=str, help='model architecture')
parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
parser.add_argument('--save-epoch', default=5, type=int, help='save the model every save_epoch')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lri', '--learning-rate-image', default=0.005, type=float, help='initial learning rate for image encoder')
parser.add_argument('--lrt', '--learning-rate-text', default=0.0005, type=float, help='initial learning rate for text encoder')
parser.add_argument('--wdi', '--weight-decay-image', default=0.1, type=float, help='weight decay for image encoder (default: 0.1)')
parser.add_argument('--wdt', '--weight-decay-text', default=0.1, type=float, help='weight decay for text encoder (default: 0.1)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: False)')
parser.add_argument("--warmup_length", type=int, default=500, help='initial warmup iterations for the scheduler (default: 500)')

parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for the image reprogramming perturbation (default: 0.0)')
parser.add_argument('--image-resolution', default=64, type=int, help='preprocessing image resolution (default: 64)')
parser.add_argument('--up-resolution', default=224, type=int, help='resolutions used for upsampling in model reprogramming (default: 224)')
parser.add_argument('--mr-resolution', default=192, type=int, help='resolution used in model reprogramming (default: 192)')
parser.add_argument('--print-freq', '-p', default=150, type=int, help='print frequency (default: 150)')
parser.add_argument('--random-seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--load', default=None, type=str, help='name of experiment being loaded')
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
import torchvision.transforms as transforms
from utils import ImageNet, PCAM, FocalLoss, cosine_lr
from utils.label_map import cifar10_labels, cifar10_classnames, cifar100_classnames, imagenet_classnames

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


def main():
    # Setting up data augmentation if needed
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
        labels = cifar10_labels
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        classnames = cifar100_classnames
        labels = cifar10_labels
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == 'ImageNet':
        classnames = imagenet_classnames
        labels = cifar10_labels
        train_loader = torch.utils.data.DataLoader(
            ImageNet(root='/nobackup-fast/ageng/datasets/ImageNet-1k/', train=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            ImageNet('/nobackup-fast/ageng/datasets/ImageNet-1k/', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "PCAM":
        train_loader = torch.utils.data.DataLoader(
            PCAM(path='/nobackup-slow/dataset/PCAM', train=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            PCAM(path='/nobackup-slow/dataset/PCAM', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # Loading zero-shot CLIP model
    model, preprocess = clip.load('ViT-B/32', mr_resolution=args.mr_resolution, up_resolution=args.up_resolution, dropout=args.dropout)

    # Choose to load from pre-trained checkpoint
    if args.load is not None:
        checkpoint = torch.load("/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_5_resolution_192.pth".format(in_dataset=args.in_dataset, name=args.load))
        model.load_state_dict(checkpoint['state_dict'].module.state_dict())

    # Issue -> https://github.com/openai/CLIP/issues/40
    model = model.float()

    # Reinitializing reprogramming parameters
    if args.load is None:
        model.reinitialize_mr_parameters(args.mr_resolution, reinitialize_image=True, reinitialize_text=False)
    
    # Specifying gpu usage for model
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # Get Model Repogramming image input shift parameter
    mr_image = [p for n, p in list(model.named_parameters()) if 'module.mr_image' in n]

    # Get Model Repogramming text input shift parameter
    include = lambda n : "module.token_embedding" in n or "module.positional_embedding" in n
    mr_text = [p for n, p in list(model.named_parameters()) if include(n) and p.requires_grad]

    # Defining image and text loss function 
    image_loss = FocalLoss(alpha=0.8, gamma=4.0, reduction='mean', eps=1e-7).cuda()
    text_loss = FocalLoss(alpha=0.8, gamma=4.0, reduction='mean', eps=1e-7).cuda()
    # image_loss = torch.nn.CrossEntropyLoss().cuda()
    # text_loss = torch.nn.CrossEntropyLoss().cuda()
    
    # Defining Adam optimizers
    optimizer_image = torch.optim.Adam(mr_image, lr=args.lri, weight_decay=args.wdi)
    optimizer_text = torch.optim.AdamW(mr_text, lr=args.lrt, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wdt)

    # Defining learning rate scheduler
    scheduler_image = cosine_lr(optimizer_image, args.lri, args.warmup_length, args.epochs*len(train_loader))
    scheduler_text = cosine_lr(optimizer_text, args.lrt, args.warmup_length, args.epochs*len(train_loader))

    # Iterating through training epochs
    for epoch in range(0, args.epochs):

        # Train model for one epoch
        train(train_loader, model, preprocess, image_loss, text_loss, optimizer_image, 
                optimizer_text, scheduler_image, scheduler_text, classnames, labels, epoch)

        # Evaluate model on testing dataset
        accuracy = validate(val_loader, model, preprocess, image_loss, classnames, labels, epoch)

        # Save the checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model,
                'accuracy': accuracy,
                'mr_resolution': args.mr_resolution,
                'up_resolution' : args.up_resolution,
            }, args.mr_resolution, epoch + 1)


def train(train_loader, model, preprocess, image_loss, text_loss, optimizer_image, optimizer_text, scheduler_image, scheduler_text, classnames, labels, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()

    # Switch model to train mode
    model.train()

    end = time.time()
    num_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Update learning rate with scheduler
        scheduler_image(i + epoch*num_batches)
        scheduler_text(i + epoch*num_batches)

        # Tokenize targets to CLIP expected values
        texts = torch.cat([clip.tokenize(labels[0](classnames[target])) for target in targets]).cuda()

        # Forward through CLIP vision and text encoder
        image_features, text_features, logit_scale = model(inputs, texts)
        
        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        image_logits = logit_scale * image_features @ text_features.t()
        text_logits = logit_scale * text_features @ image_features.t()

        # Calculate the loss function for CLIP
        ground_truth = torch.arange(len(image_logits)).long().cuda()
        loss = (image_loss(image_logits, ground_truth) + text_loss(text_logits, ground_truth)) / 2
        
        # Backpropagate gradients and update model
        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(mr_text, 1.0)
        optimizer_image.step()
        optimizer_text.step()

        # Record the loss and elapsed time
        batch_loss.update(loss.data, inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Current Learning Rates | Image: {lri} | Text: {lrt}'.format(lri=get_lr(optimizer_image), lrt=get_lr(optimizer_text)))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, 
                      loss=batch_loss))
    
    print('---------------> Training Loss {loss.avg:.3f} <---------------'.format(loss=batch_loss))


def validate(val_loader, model, preprocess, image_loss, classnames, labels, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to evaluation mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Tokenize targets to CLIP expected values
        texts = torch.cat([clip.tokenize(labels[0](classname)) for classname in classnames]).cuda()

        # Forward through CLIP vision and text encoder
        with torch.no_grad():
            image_features, text_features, logit_scale = model(inputs, texts)

        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        similarity = logit_scale * image_features @ text_features.t()

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


def save_checkpoint(state, resolution, epoch):
    directory = "/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = directory + 'checkpoint_{epoch}_resolution_{res}.pth'.format(epoch=epoch, res=resolution)
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