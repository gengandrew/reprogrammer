import os
import time
import argparse

parser = argparse.ArgumentParser(description='Pytorch CLIP Out-of-distribution detection')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='ViT-B/32', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=50, type=int, help='mini-batch size')

parser.add_argument('--up-resolution', default=224, type=int, help='resolutions used for upsampling in model reprogramming (default: 224)')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for the image reprogramming perturbation (default: 0.0)')
parser.add_argument('--image-resolution', default=128, type=int, help='preprocessing image resolution (default: 64)')
parser.add_argument('--mr-resolution', '-res', default=192, type=int, help='resolution used in model reprogramming')
parser.add_argument('--random_seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--augment', default='True', type=str, help='whether augmentation is used')
parser.add_argument('--method', default='base', type=str, help='scoring function')
parser.add_argument('--name', required=True, type=str, help='the name of the model trained')

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
import torch.nn as nn
from utils import ImageNet
import torch.nn.functional as F
import utils.svhn_loader as svhn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from clip.ftmodel import ImageEncoder, ImageClassifier, LogisticRegressionHead, get_classification_head
from utils.label_map import cifar10_labels, cifar10_classnames, cifar100_classnames, imagenet_classnames

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_msp_score(model, inputs, texts):
    with torch.no_grad():
        image_features, text_features, logit_scale = model(inputs, texts)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
    scores = torch.max(similarity, dim=1)[0].numpy()

    return scores


def get_lp_score(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores


def get_score(model, preprocess, inputs, texts, method):
    if method == 'lp' or method == 'ft':
        scores = get_lp_score(model, inputs)
    else:
        scores = get_msp_score(model, inputs, texts)
    
    return scores


def get_accuracy_metrics(model, inputs, texts, method):
    if method == 'lp' or method == 'ft':
        with torch.no_grad():
            outputs = model(inputs)
        
        outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()
        predictions = np.argmax(outputs, axis=1)
        confidences = np.max(outputs, axis=1)
    else:
        with torch.no_grad():
            image_features, text_features, logit_scale = model(inputs, texts)

        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        similarity = logit_scale * image_features @ text_features.t()
        similarity = similarity.softmax(dim=-1).detach().cpu().numpy()
        predictions = np.argmax(similarity, axis=1)
        confidences = np.max(similarity, axis=1)
    
    return predictions, confidences


def eval_ood_detector(args, directory, out_datasets):
    if args.augment == 'True':
        transform = transforms.Compose([
            transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_resolution),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_resolution),
            convert_image_to_rgb,
            transforms.ToTensor()
        ])

    # Setting up In-distribution dataset
    if args.in_dataset == "CIFAR-10":
        classnames = cifar10_classnames
        labels = cifar10_labels
        ID_dataset = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        ID_loader = torch.utils.data.DataLoader(ID_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.in_dataset == "CIFAR-100":
        classnames = cifar100_classnames
        labels = cifar10_labels
        ID_dataset = datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        ID_loader = torch.utils.data.DataLoader(ID_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.in_dataset == 'ImageNet':
        classnames = imagenet_classnames
        labels = cifar10_labels
        ID_dataset = ImageNet('/nobackup-fast/ageng/datasets/ImageNet-1k/', train=False, transform=transform)
        ID_loader = torch.utils.data.DataLoader(ID_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Loading tokenized CLIP preprocessed texts
    texts = torch.cat([clip.tokenize(labels[0](classname)) for classname in classnames]).cuda()

    # Loading CLIP models
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
    elif args.method == 'rp':
        model, preprocess = clip.load('ViT-B/32', mr_resolution=args.mr_resolution, up_resolution=args.up_resolution, dropout=args.dropout)
        checkpoint = torch.load("/nobackup-fast/ageng/checkpoints/{in_dataset}/{name}/checkpoint_{epochs}_resolution_192.pth".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
        model.load_state_dict(checkpoint['state_dict'].module.state_dict())
    
    model = model.cuda()
    model.eval()
    if ',' in args.gpu:
        model = nn.DataParallel(model)

    print("Processing in-distribution images")
    count = 0
    initial_time = time.time()
    score_file = open(os.path.join(directory, "in_scores.txt"), 'w')
    prediction_file = open(os.path.join(directory, "in_labels.txt"), 'w')
    for i, (inputs, targets) in enumerate(ID_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        scores = get_score(model, preprocess, inputs, texts, args.method)
        for score in scores:
            score_file.write("{}\n".format(score))
        
        predictions, confidences = get_accuracy_metrics(model, inputs, texts, args.method)

        for k in range(predictions.shape[0]):
            prediction_file.write("{} {} {}\n".format(targets[k], predictions[k], confidences[k]))

        count += inputs.shape[0]
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, len(ID_loader.dataset), time.time()-initial_time))
        initial_time = time.time()

    score_file.close()

    for out_dataset in out_datasets:
        if out_dataset == 'SVHN':
            OOD_dataset = svhn.SVHN('/u/a/g/ageng/Research/clip/datasets/ood_datasets/svhn/', split='test', download=False, transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'dtd':
            OOD_dataset = datasets.ImageFolder(root="/nobackup-slow/dataset/dtd/images", transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'places365':
            OOD_dataset = datasets.ImageFolder(root="/nobackup-slow/dataset/places365", transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'iNaturalist':
            OOD_dataset = datasets.ImageFolder(root="/nobackup-fast/ageng/datasets/iNaturalist", transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'SUN':
            OOD_dataset = datasets.ImageFolder(root="/nobackup-fast/ageng/datasets/SUN", transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'Places':
            OOD_dataset = datasets.ImageFolder(root="/nobackup-fast/ageng/datasets/Places", transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        else:
            OOD_dataset = datasets.ImageFolder("/nobackup-slow/dataset/{}".format(out_dataset), transform=transform)
            OOD_loader = torch.utils.data.DataLoader(OOD_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        print("Processing out-of-distribution images")
        ood_directory = os.path.join(directory, out_dataset)
        if not os.path.exists(ood_directory):
            os.makedirs(ood_directory)

        ood_score_file = open(os.path.join(ood_directory, "out_scores.txt"), 'w')
        initial_time = time.time()
        count = 0
        for j, data in enumerate(OOD_loader):
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()

            scores = get_score(model, preprocess, inputs, texts, args.method)
            for score in scores:
                ood_score_file.write("{}\n".format(score))

            count += inputs.shape[0]
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, len(OOD_loader.dataset), time.time()-initial_time))
            initial_time = time.time()
        
        ood_score_file.close()


if __name__ == '__main__':
    directory = "outputs/{in_dataset}/{method}/{name}/".format(in_dataset=args.in_dataset, method=args.method, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # out_datasets = ['SVHN', 'LSUN_C', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    out_datasets = ['iNaturalist', 'SUN', 'Places', 'dtd']

    eval_ood_detector(args, directory, out_datasets)