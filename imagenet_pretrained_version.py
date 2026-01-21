import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

from resnet import *

"""
NOTE:
    Only for pretrained_model PTQ
"""

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

""" Modified: Add args_file """
import argparse
import yaml
import munch
from pathlib import Path

def merge_nested_dict(d, other):
    new = dict(d)
    for k, v in other.items():
        if d.get(k, None) is not None and type(v) is dict:
            new[k] = merge_nested_dict(d[k], v)
        else:
            new[k] = v
    return new
def get_config(default_file):
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    arg = p.parse_args()

    with open(default_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for f in arg.config_file:
        if not os.path.isfile(f):
            raise FileNotFoundError('Cannot find a configuration file at', f)
        with open(f) as yaml_file:
            c = yaml.safe_load(yaml_file)
            cfg = merge_nested_dict(cfg, c)

    return munch.munchify(cfg)

""" Modified """
def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    # --------------- ImageNet-val: X + Quant ---------------
    from torch.utils.data import SequentialSampler, Subset, DataLoader

    imagenet_datapath = "/home/wangzy/project/DATA/ILSVRC2012_img_val" # /path/to/imagenet
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_set = datasets.ImageFolder( root=imagenet_datapath,
                                     transform=test_transform )

    N_calibration = 2000
    indices = random.sample( range(len(test_set)), N_calibration )
    calibration_subset = Subset( test_set, indices )
    calibration_loader = DataLoader( dataset=test_set,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=False )

    test_loader = DataLoader( dataset=test_set,
                              batch_size=eval_batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=False )

    return calibration_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


# def train_model(model,
#                 train_loader,
#                 test_loader,
#                 device,
#                 learning_rate=1e-1,
#                 num_epochs=200):
# 
#     # The training configurations were not carefully selected.
# 
#     criterion = nn.CrossEntropyLoss()
# 
#     model.to(device)
# 
#     # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
#     optimizer = optim.SGD(model.parameters(),
#                           lr=learning_rate,
#                           momentum=0.9,
#                           weight_decay=1e-4)
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                      milestones=[100, 150],
#                                                      gamma=0.1,
#                                                      last_epoch=-1)
#     # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 
#     # Evaluation
#     model.eval()
#     eval_loss, eval_accuracy = evaluate_model(model=model,
#                                               test_loader=test_loader,
#                                               device=device,
#                                               criterion=criterion)
#     print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
#         -1, eval_loss, eval_accuracy))
# 
#     for epoch in range(num_epochs):
# 
#         # Training
#         model.train()
# 
#         running_loss = 0
#         running_corrects = 0
# 
#         for inputs, labels in train_loader:
# 
#             inputs = inputs.to(device)
#             labels = labels.to(device)
# 
#             # zero the parameter gradients
#             optimizer.zero_grad()
# 
#             # forward + backward + optimize
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
# 
#         train_loss = running_loss / len(train_loader.dataset)
#         train_accuracy = running_corrects / len(train_loader.dataset)
# 
#         # Evaluation
#         model.eval()
#         eval_loss, eval_accuracy = evaluate_model(model=model,
#                                                   test_loader=test_loader,
#                                                   device=device,
#                                                   criterion=criterion)
# 
#         # Set learning rate scheduler
#         scheduler.step()
# 
#         print(
#             "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
#             .format(epoch, train_loss, train_accuracy, eval_loss,
#                     eval_accuracy))
# 
#     return model


def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


# def create_model(pretrained=False, num_classes=10):
def create_model( args ):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    # model = resnet18(num_classes=num_classes, pretrained=False)
    """ Modified """
    model = None
    if args.arch == 'resnet18':
        model = resnet18( pretrained=args.pre_trained )
    elif args.arch == 'resnet50':
        model = resnet50( pretrained=args.pre_trained )

    if model is None:
        logger.error('Model arch `%s` is not defined' % (args.arch))

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model


""" Modified: Add fuse_model_18/50 """
def fuse_model_18( model ):
    model = torch.quantization.fuse_modules( model,
                                             [["conv1", "bn1", "relu"]],
                                             inplace=True)
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block,
                    [["conv1", "bn1", "relu1"],
                     ["conv2", "bn2"]],
                    inplace=True
                )
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block,
                                                        [["0", "1"]],
                                                        inplace=True)
    return model

def fuse_model( model ):
    from torch.quantization import fuse_modules

    model = fuse_modules( model,
                          [["conv1", "bn1", "relu"]],
                          inplace=True)

    for module_name, module in model.named_children():
        if "layer" in module_name:
            for block_name, block in module.named_children():
                print(block_name, "\n", block)
                fuse_list = []

                if all(hasattr(block, attr) for attr in ["conv1", "bn1", "relu1"]):
                    print("  conv1")
                    fuse_list.append(["conv1", "bn1", "relu1"])
                if all(hasattr(block, attr) for attr in ["conv2", "bn2", "relu2"]):
                    print("  conv2")
                    fuse_list.append(["conv2", "bn2"])
                    # fuse_list.append(["conv2", "bn2", "relu2"])
                if all(hasattr(block, attr) for attr in ["conv3", "bn3"]):
                    print("  conv3")
                    fuse_list.append(["conv3", "bn3"])

                print(fuse_list)
                if fuse_list:
                    fuse_modules(block, fuse_list, inplace=True)
                
                if hasattr( block, "downsample" ) and block.downsample is not None:
                    fuse_modules( block.downsample,
                                  [["0", "1"]],
                                  inplace=True )
    return model

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def model_equivalence(model_1,
                      model_2,
                      device,
                      rtol=1e-05,
                      atol=1e-08,
                      num_tests=100,
                      input_size=(1, 3, 32, 32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol,
                       equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def main():
    script_dir= Path.cwd()
    args = get_config( default_file=script_dir / 'config.yaml' )

    random_seed = 0
    num_classes = 10
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = "resnet18_pretrained.pt"
    quantized_model_filename = "resnet18_quantized_pretrained_2.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    """ Modified """
    # use pretrained model for PTQ.
    model = create_model( args )
    # model = resnet18( pretrained=args.pre_trained )
    train_loader, test_loader = prepare_dataloader(num_workers=8,
                                                   train_batch_size=128,
                                                   eval_batch_size=256)

    """ Modified """
    # Delete Train model.

    # # Save model.
    # save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # # Load a pretrained model.
    # model = load_model( model=model,
    #                     model_filepath=model_filepath,
    #                     device=cpu_device )

    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    model.eval()
    # The model has to be switched to evaluation mode before any layer fusion.
    # Otherwise the quantization will not work correctly.
    fused_model.eval()

    # Fuse the model in place rather manually.
    print("\n", "="*20, "\nFuse model ...")
    print(isinstance(model.layer1[0], torchvision.models.resnet.BasicBlock))
    fused_model = fuse_model( fused_model )

    # Print FP32 model.
    print("\n", "="*40, "\nFP32 Model print:")
    print(model)
    # Print fused model.
    print("\n", "="*40, "\nFused Model print:")
    print(fused_model)

    # Model and fused model should be equivalent.
    print("\nassert FP32 Model and Fused Model is equivalent or not:")
    assert model_equivalence(
        model_1=model,
        model_2=fused_model,
        device=cpu_device,
        rtol=1e-03,
        atol=1e-06,
        num_tests=100,
        input_size=(
            1, 3, 32,
            32)), "Fused model is not equivalent to the original model!"

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedResNet18(model_fp32=fused_model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.
    # quantized_model = QuantizedResNet18(model_fp32=model)

    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config
    print("\nQConfig:")
    print(quantized_model.qconfig)

    # https://pytorch.org/docs/master/torch.quantization.html#torch.quantization. prepare
    torch.quantization.prepare(quantized_model, inplace=True)

    # Use training data for calibration.
    calibrate_model(model=quantized_model,
                    loader=train_loader,
                    device=cpu_device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

    quantized_model.eval()

    # Print quantized model.
    print("\n", "="*40, "\nQuantized Model print:")
    print(quantized_model)

    # Save quantized model.
    print("\n", "="*40, "\nSave and Load quantized_model ...")
    save_torchscript_model(model=quantized_model,
                           model_dir=model_dir,
                           model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(
        model_filepath=quantized_model_filepath, device=cpu_device)

    print("\n", "="*40, "\nEval ...")
    _, fp32_eval_accuracy = evaluate_model( model=model,
                                            test_loader=test_loader,
                                            device=cpu_device,
                                            criterion=None )
    # 换成quantized_model会不会得到eval结果, quantized_model和quantized_jit_model有什么区别
    _, int8_eval_accuracy = evaluate_model( model=quantized_model,
                                            test_loader=test_loader,
                                            device=cpu_device,
                                            criterion=None )
    _, int8_jit_eval_accuracy = evaluate_model( model=quantized_jit_model,
                                                test_loader=test_loader,
                                                device=cpu_device,
                                                criterion=None )

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
    print("INT8 JIT evaluation accuracy: {:.3f}".format(int8_jit_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cpu_device,
                                                           input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(
        model=quantized_model,
        device=cpu_device,
        input_size=(1, 3, 32, 32),
        num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(
        model=quantized_jit_model,
        device=cpu_device,
        input_size=(1, 3, 32, 32),
        num_samples=100)

    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(
        fp32_cpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(
        int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(
        int8_jit_cpu_inference_latency * 1000))


if __name__ == "__main__":

    main()
