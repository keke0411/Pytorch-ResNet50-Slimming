import argparse
import torch
import torch.nn as nn
import torchsummary
from model import resnet50
from common import create_dataloader

def arg(parser):
    parser.add_argument('--bz', type=int, default=128, help='The batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='The device to train')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--prune_model_path', type=str, default='./save/pruned_model.pth', help='The path to load pruned model')
    parser.add_argument('--save_path', type=str, default='./save/prune_model_finetune.pth', help='The path to save model' )
    parser.add_argument('--train_path', type=str, default='./dataset/sign_mnist_train.csv', help='The path to train data')
    parser.add_argument('--test_path', type=str, default='./dataset/sign_mnist_test.csv', help='The path to test data')
    args = parser.parse_args()
    return args

def train(opt, model, train_loader, test_loader):
    best_acc = 0
    for epoch in range(opt.n_epochs):
        model.train()
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(opt.device), label.to(opt.device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print('Epoch: {:03d}/{:03d} | Batch {:03d}/{:03d} | Loss: {:.5f}'.format(
                epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss.item()
            ))
        acc = test(opt, model, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), opt.save_path)
            print('Saving model with acc {:.5f}'.format(best_acc))

def test(opt, model, test_loader):
    model.eval()
    correct = 0
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(opt.device), label.to(opt.device)
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print('Test accuracy: {:.5f}'.format(acc))
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Sign Language MNIST Pruned Model Finetune')
    opt = arg(parser)

    # set device
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # 保證輸入相同時每次運算結果一樣
    torch.manual_seed(opt.seed)  # 固定隨機種子

    # create dataloader
    train_loader, test_loader = create_dataloader(opt)

    # load model
    model = None
    save_dict = torch.load(opt.prune_model_path)
    model = resnet50(cfg=save_dict['cfg']).to(opt.device)
    model.load_state_dict(save_dict['state_dict'])
    # torchsummary.summary(model, (1, 28, 28))
    # print(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # create loss function
    criterion = nn.CrossEntropyLoss()

    # train
    train(opt, model, train_loader, test_loader)