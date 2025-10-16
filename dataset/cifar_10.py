import torchvision
import torchvision.transforms as transforms

def load_cifar10(data_dir="./data"):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train, test
