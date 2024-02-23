import torch
import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    
    assert len(base_lrs) == len(optimizer.param_groups)
    
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            
            assign_learning_rate(param_group, lr)
    
    return _lr_adjuster


def step_lr(optimizer, base_lrs, steps):
    lr_schedule = [0.4*steps, 0.7*steps, 0.8*steps]
    lr_rates = [0.5*base_lrs, 0.2*base_lrs, 0.1*base_lrs]

    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            lr = base_lr
            if step >= lr_schedule[0]:
                lr = lr_rates[0]
    
            if step >= lr_schedule[1]:
                lr = lr_rates[1]
            
            if step >= lr_schedule[2]:
                lr = lr_rates[2]
            
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def polynomial_lr(optimizer, start_lr, end_lr, total_steps, power=1):
    start_lr = start_lr - end_lr
    
    if not isinstance(start_lr, list):
        base_lrs = [start_lr for _ in optimizer.param_groups]
    
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            lr = (base_lr * ((1 - (float(step)/total_steps))**power)) + end_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster