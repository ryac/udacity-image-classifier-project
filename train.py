# imports..
# from workspace_utils import active_session
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from datetime import datetime
import argparse
import sys

device = None


def init():

    dataloaders, class_to_idx = getDataloaders(args.opts[0])

    device = args.device
    print('device: {}'.format(device))

    print('model will be saved to: {}'.format(args.save_dir))

    # load pre-trained model..
    if (args.arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif (args.arch == 'resnet101'):
        model = models.resnet101(pretrained=True)
        in_features = 2048
    else:
        print('model name in --arch not supported.')
        sys.exit()

    # turn off gradients in pre-trained model..
    for param in model.parameters():
        param.requires_grad = False

    # build new classifier..
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, args.hidden_units[0])),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=args.dropout[0])),
        ('fc2', nn.Linear(args.hidden_units[0], args.hidden_units[1])),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=args.dropout[1])),
        ('fc3', nn.Linear(args.hidden_units[1], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # add classifier to model and set optimizer..
    if (args.arch == 'vgg16'):
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(),
                               lr=args.learning_rate)
    else:
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(),
                               lr=args.learning_rate)

    # set criterion..
    criterion = nn.NLLLoss()

    model.to(device)

    # training..
    with active_session():
        epochs = args.epochs
        steps = 0
        running_loss = 0
        print_every = 5
        print('>> training start..')
        for epoch in range(epochs):
            for inputs, labels in dataloaders['train']:
                steps += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in dataloaders['valid']:

                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch + 1} / {epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Test loss: {test_loss / len(dataloaders['valid']):.3f}.. "
                          f"Test accuracy: {accuracy / len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    model.train()

    # save model..
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }

    checkpoint_file = buildCheckpointFile(args.save_dir)
    print('DONE! saving model to {}'.format(checkpoint_file))
    torch.save(checkpoint, checkpoint_file)


def getDataloaders(dir):
    # data directories..
    train_dir = dir + '/train'
    valid_dir = dir + '/valid'
    test_dir = dir + '/test'

    # data transformations..
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # load the ImageFolder and datasets..
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    return dataloaders, image_datasets['train'].class_to_idx


def buildCheckpointFile(directory=None):
    directory = directory if directory is not None else '.'
    directory = directory.rstrip('/')

    filename = '{}-{}-e{}-lr{}.pth'.format(
        datetime.utcnow().strftime('%Y-%m-%dT%H%M%S'), args.arch, args.epochs, args.learning_rate)
    checkpoint_file = '{}/{}'.format(directory, filename)
    return checkpoint_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a model..')

    parser.add_argument('opts', nargs=1, type=str,
                        help='The directory of where training/validation/test directories are in.')
    parser.add_argument('--save', '-s', dest='save_dir',
                        help='Save models in this directory (eg: checkpoints)')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda',
                        default='cpu', help='Pass flag if you want to use the GPU, defaults to CPU.')
    parser.add_argument('--arch', dest='arch', default='vgg16',
                        help='Load a pre-trained model from Pytorch (default=vgg16 [vgg16 | resnet101]).')
    parser.add_argument('--epochs', dest='epochs', default=2, type=int,
                        help='Number of epochs when training (default=2).')
    parser.add_argument('--hidden_units', '-hu', nargs=2, type=int, default=[1280, 256],
                        help='Hidden units for the hidden layers (default=[1280, 256]).')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='Set the learning rate (default=0.001).')
    parser.add_argument('--dropout', nargs=2, type=float,
                        default=[0.4, 0.2], help='Dropout probability (default=[0.4, 0.2]).')

    args = parser.parse_args()

    # print('hidden_units:', args.hidden_units)
    # print('dropout:', args.dropout)
    # print('learning_rate:', args.learning_rate)
    # print('label_mapping:', args.label_mapping)
    # print(buildCheckpointFile(args.save_dir))
    init()

    # example:
    # python train.py flowers --gpu
    # python train.py flowers --gpu -s models --epochs 1
    # python train.py flowers --gpu --arch vgg16 --epochs 1 -hu 1280 512 -lr 0.002 --dropout 0.2 0.2
