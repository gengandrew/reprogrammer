import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch CLIP Model Reprogramming OOD Robustness Evaluation')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--ood-dataset', default="CIFAR-10.1", type=str, help='out-of-distribution dataset')
parser.add_argument('--model-arch', default='ViT-B/32', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=75, type=int, help='print frequency (default: 75)')
parser.add_argument('--random-seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--method', default="base", type=str, help='The methodology used to fine-tune')

parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for the image reprogramming perturbation (default: 0.0)')
parser.add_argument('--alpha', default=0.0, type=float, help='weight of the zero-shot residual connection')
parser.add_argument('--image-resolution', default=128, type=int, help='preprocessing image resolution (default: 128)')
parser.add_argument('--up-resolution', default=224, type=int, help='resolutions used for upsampling in model reprogramming (default: 224)')
parser.add_argument('--mr-resolution', default=192, type=int, help='resolution used in model reprogramming (default: 192)')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.set_defaults(augment=True)

# Grabbing cli arguments and printing results
args = parser.parse_args()
print_args = '*'*45
for key,value in args._get_kwargs():
    print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

print_args = print_args + '\n' + '*'*45
print(print_args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import clip
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils import CIFAR101, STL10, ImageNet, ImageNetV2, ImageNetA, ImageNetR, ImageNetS
from clip.ftmodel import ImageEncoder, ImageClassifier, LogisticRegressionHead, get_classification_head
from utils.label_map import cifar10_labels, cifar10_classnames, imagenet_classnames, imagenet_a_indices, imagenet_r_indices

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

    # Setting up robustness dataset for evaluation
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.ood_dataset == "CIFAR-10":
        classnames = cifar10_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "CIFAR-10.1":
        classnames = cifar10_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            CIFAR101(root='./datasets/cifar10.1', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "STL10":
        classnames = cifar10_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            STL10(root='./datasets/stl10', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == 'ImageNet':
        classnames = imagenet_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            ImageNet('/nobackup-fast/ageng/datasets/ImageNet-1k/', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "ImageNetV2":
        classnames = imagenet_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            ImageNetV2(root='/nobackup-fast/ageng/datasets/ImageNetV2', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "ImageNet-A":
        classnames = imagenet_classnames
        labels = cifar10_labels
        label_map = imagenet_a_indices
        test_loader = torch.utils.data.DataLoader(
            ImageNetA(root='/nobackup-fast/ageng/datasets/ImageNet-A', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "ImageNet-R":
        classnames = imagenet_classnames
        labels = cifar10_labels
        label_map = imagenet_r_indices
        test_loader = torch.utils.data.DataLoader(
            ImageNetR(root='/nobackup-fast/ageng/datasets/ImageNet-R', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.ood_dataset == "ImageNet-Sketch":
        classnames = imagenet_classnames
        labels = cifar10_labels
        label_map = None
        test_loader = torch.utils.data.DataLoader(
            ImageNetS(root='/nobackup-fast/ageng/datasets/ImageNet-Sketch', transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # Loading reprogrammed CLIP model remote directory [/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth]
    if args.method == 'base':
        model, preprocess = clip.load('ViT-B/32')
    elif args.method == 'lp':
        image_encoder = ImageEncoder('ViT-B/32', keep_lang=True)
        classification_head = LogisticRegressionHead(input_size=512, output_size=len(classnames), normalize=True)
        delattr(image_encoder.model, 'transformer') #TODO perhaps we don't need to remove this param
        model = ImageClassifier(image_encoder, classification_head)
        checkpoint = torch.load("/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
        model.load_state_dict(checkpoint['state_dict'].module.state_dict())
        preprocess = model.preprocess
    elif args.method == 'ft':
        image_encoder = ImageEncoder('ViT-B/32', keep_lang=True)
        classification_head = get_classification_head(image_encoder.model, classnames, cifar10_labels)
        delattr(image_encoder.model, 'transformer') #TODO perhaps we don't need to remove this param
        model = ImageClassifier(image_encoder, classification_head)
        checkpoint = torch.load("/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
        model.load_state_dict(checkpoint['state_dict'].module.state_dict())
        preprocess = model.preprocess
    elif args.method == 'rp' or args.method == 'resrp':
        model, preprocess = clip.load('ViT-B/32', mr_resolution=args.mr_resolution, up_resolution=args.up_resolution, dropout=args.dropout)
        checkpoint = torch.load("/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_{epochs}_resolution_{res}.pth".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs, res=args.mr_resolution))
        model.load_state_dict(checkpoint['state_dict'].module.state_dict())

    # Issue -> https://github.com/openai/CLIP/issues/40
    model = model.float()
    
    # Specifying gpu usage for model
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # Defining image loss function
    image_loss = torch.nn.CrossEntropyLoss().cuda()

    # Evaluating the Out-of-distribution dataset for robustness
    if args.method == 'base':
        accuracy = evaluate_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=label_map)
    elif args.method == 'lp' or args.method == 'ft':
        accuracy = evaluate_lp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=label_map)
    elif args.method == 'rp':
        accuracy = evaluate_rp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=label_map)
    elif args.method == 'resrp':
        accuracy = evaluate_resrp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=label_map)


def evaluate_resrp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=None):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Loading in the zero-shot model
    zero_shot, _ = clip.load('ViT-B/32')

    # Switch model to evaluation mode
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Tokenize targets to CLIP expected values
        zs_inputs = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in inputs], dim=0).cuda()
        texts = torch.cat([clip.tokenize(labels[0](classname)) for classname in classnames]).cuda()

        # Forward through CLIP vision and text encoder
        with torch.no_grad():
            zs_image_features, zs_text_features, zs_logit_scale = zero_shot(zs_inputs, texts)
            image_features, text_features, logit_scale = model(inputs, texts)

        # Linear averaging between RP and ZS features
        image_features = (1-args.alpha)*image_features + args.alpha*zs_image_features
        text_features = (1-args.alpha)*text_features + args.alpha*zs_text_features

        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        similarity = logit_scale * image_features @ text_features.t()

        # Redefine targets in terms of ImageNet-1k labels
        if label_map is not None:
            targets = torch.Tensor([label_map[target.item()] for target in targets]).type(torch.long).cuda()

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
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Robustness Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def evaluate_lp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=None):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Loading in the classnames for dataset
    test_classnames = test_loader.dataset.classnames

    # Switch model to evaluation mode
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Tokenize targets to CLIP expected values
        inputs = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in inputs], dim=0).cuda()

        # Forward through CLIP vision and text encoder
        similarity = model(inputs)

        # Redefine targets in terms of ImageNet-1k labels
        if label_map is not None:
            targets = torch.Tensor([label_map[target.item()] for target in targets]).type(torch.long).cuda()

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
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Robustness Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def evaluate_rp_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=None):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Loading in the classnames for dataset
    test_classnames = test_loader.dataset.classnames

    # Switch model to evaluation mode
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
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

        # Redefine targets in terms of ImageNet-1k labels
        if label_map is not None:
            targets = torch.Tensor([label_map[target.item()] for target in targets]).type(torch.long).cuda()

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
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Robustness Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def evaluate_robustness(test_loader, model, preprocess, image_loss, classnames, labels, label_map=None):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Loading in the classnames for dataset
    test_classnames = test_loader.dataset.classnames

    # Switch model to evaluation mode
    model.eval()
    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Redefining inputs and targets to CLIP expected values
        inputs = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in inputs], dim=0).cuda()
        texts = torch.cat([clip.tokenize(labels[0](classname)) for classname in classnames]).cuda()

        # Forward through CLIP vision and text encoder
        with torch.no_grad():
            image_features, text_features, logit_scale = model(inputs, texts)

        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        similarity = logit_scale * image_features @ text_features.t()

        # Redefine targets in terms of ImageNet-1k labels
        if label_map is not None:
            targets = torch.Tensor([label_map[target.item()] for target in targets]).type(torch.long).cuda()

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
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Robustness Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def convert_image_to_rgb(image):
    return image.convert("RGB")


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