import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import transformers
from art.estimators.classification.hugging_face import HuggingFaceClassifierPyTorch
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import insert_image

def poison_data(data_path, backdoor_path, samples_per_class=100, poison_percent=0.5):
    """ Load and poison data with provided backdoor image """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        ])

    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    
    labels = np.asarray(train_dataset.targets)
    classes = np.unique(labels)
    
    x_subset = []
    y_subset = []
    
    for c in classes:
        indices = np.where(labels == c)[0][:samples_per_class]
        for i in indices:
            x_subset.append(train_dataset[i][0])
            y_subset.append(train_dataset[i][1])
    
    x_subset = np.stack(x_subset)
    y_subset = np.asarray(y_subset)

    poison_func = lambda x: insert_image(
        x,
        backdoor_path=backdoor_path,
        channels_first=True,
        random=False,
        x_shift=0,
        y_shift=0,
        size=(32, 32),
        mode='RGB',
        blend=0.8
    )
    
    backdoor = PoisoningAttackBackdoor(poison_func)
    
    source_class = 0
    target_class = 1
    
    x_poison = np.copy(x_subset)
    y_poison = np.copy(y_subset)
    is_poison = np.zeros(len(x_subset)).astype(bool)
    
    indices = np.where(y_subset == source_class)[0]
    num_poison = int(poison_percent * len(indices))
    
    for i in indices[:num_poison]:
        x_poison[i], _ = backdoor.poison(x_poison[i], [])
        y_poison[i] = target_class
        is_poison[i] = True
    return x_poison, y_poison, is_poison


def load_model(model_path, num_labels=None):
    """ Load HuggingFace model and prepare for fine-tuning 
    
    Args:
        model_path: Path or name of the Hugging Face model
        num_labels: Number of output classes. If None, uses original model's number of classes (e.g., 1000 for ImageNet)
    """
    if num_labels is None:
        # Load model with original number of classes
        model = transformers.AutoModelForImageClassification.from_pretrained(model_path)
        num_labels = model.config.num_labels
    else:
        # Load model with specified number of classes (for fine-tuning/poisoning)
        model = transformers.AutoModelForImageClassification.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
            num_labels=num_labels
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    hf_model = HuggingFaceClassifierPyTorch(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_labels,
        clip_values=(0, 1),
    )
    return hf_model


def inference(x, y, hf_model):
    """ Run inference with model provided, compute accuracy """
    outputs = hf_model.predict(x)
    preds = np.argmax(outputs, axis=1)
    acc = np.mean(preds == y)
    return acc
