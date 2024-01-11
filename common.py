from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import sign_mnist

def create_dataloader(opt):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    #TODO: change the path to your own path
    train_dataset = sign_mnist(opt.train_path, transform=transform)
    test_dataset = sign_mnist(opt.test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opt.bz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader