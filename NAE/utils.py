import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import glob
import logging
import os
from natsort import ns, natsorted
from dataset_caption import imagenet_label

import torch.nn as nn
import torchvision.models as models
import numpy as np
from art.estimators.classification import PyTorchClassifier
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
import warnings

warnings.filterwarnings("ignore")



key2label = imagenet_label.refined_Label

def extract_label(path):
    """
    Given the path of the image, extract the corrsponding label,
    for the images that are not saved as npz file.
    """
    file_name = os.path.basename(path)
    label = file_name.split("_")[0]
    return label

label2key = dict((v,k) for k,v in key2label.items())



class AdvImageSet(Dataset):

    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.Resize((256,256)),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                        ])
        self.image_paths = glob.glob(os.path.join(self.root, "*"))
        self.image_paths = natsorted(self.image_paths, alg=ns.PATH)
        self.images = [self.read_image(path) for path in self.image_paths]
        self.labels = [label2key[extract_label(path)] for path in self.image_paths]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        label = torch.tensor(label).long()


        return img, label



    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) 
    
    def extract_label(self, path):
        """
        Given the path of the image, extract the corrsponding label,
        for the images that are not saved as npz file.
        """
        file_name = os.path.basename(path)
        label = file_name.split("_")[0]
        return label

def get_adv_images_loders(dir, batch_size):
    advdata = AdvImageSet(dir)
    advdata_loader = DataLoader(advdata, batch_size=batch_size, shuffle=False, num_workers=8)
    return advdata_loader


def model_selection(name):
    if name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception":
        model = models.inception_v3(pretrained=True)
    elif name == "deit-b":
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=True
        )
    elif name == "deit-s":
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True
        )
    elif name == "mixer-b":
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=True
        )
    elif name == "mixer-l":
        model = timm.create_model(
            'mixer_l16_224',
            pretrained=True
        )
    elif name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()



models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin", 'deit-b', 'deit-s',
                                'mixer-b', 'mixer-l', 'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
                                'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']





def model_transfer(data_loader, batch_size, res, logger):
    
    nb_classes = 1000
    all_acc = []
    for name in models_transfer_name:
        print(f"\n*********Transfer to {name}********")
        logger.info(f"\n*********Transfer to {name}********")

        total = 0
        correct = 0
        acc = 0
        model = model_selection(name)
        model.eval()
        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            device_type='gpu',
        )
        for i, (image, label) in enumerate(data_loader):
            image, label = image.numpy(), label.numpy()
            pred = f_model.predict(image, batch_size=batch_size)
            correct += np.sum((np.argmax(pred, axis=1) - 1) == label) if "adv" in name else np.sum(
            np.argmax(pred, axis=1) == label)
            total += label.shape[0]
        acc = correct / total


        print(f"Accuracy on adversarial examples: {acc * 100} model: {name}")
        logger.info(f"Accuracy on adversarial examples: {acc * 100} model: {name}")
        all_acc.append(acc * 100)
