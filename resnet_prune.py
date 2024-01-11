import argparse
import torch
import torch.nn as nn
import torchsummary
import numpy as np
from model import resnet50
from common import create_dataloader

def arg(parser):
    parser.add_argument('--PRUNE_PERCENT', type=float, default=0.5, help='The percentage of pruning')
    parser.add_argument('--PRUNE_PATH', type=str, default='./save/pruned_model.pth', help='The path to save pruned model')
    parser.add_argument('--PATH', type=str, default='./save/model.pth', help='The path to load model')
    parser.add_argument('--bz', type=int, default=128, help='The batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='The device to train')
    parser.add_argument('--train_path', type=str, default='./dataset/sign_mnist_train.csv', help='The path to train data')
    parser.add_argument('--test_path', type=str, default='./dataset/sign_mnist_test.csv', help='The path to test data')
    args = parser.parse_args()
    return args

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

def _get_channel_mask(PRUNE_PERCENT):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)

    threshold_index = int(total * PRUNE_PERCENT)
    threshold = y[threshold_index]

    pruned = 0
    cfg = []  #用來建立剪枝網路的CONFIG
    cfg_mask = []  #用來幫助剪枝的遮罩
    
    for name, m in model.named_modules():  
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(threshold).float().cuda()

            pruned = pruned + mask.shape[0] - torch.sum(mask)

            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

        elif 'conv3' in name:
            cfg.append(m.out_channels)   
            cfg_mask.append(torch.ones(m.weight.data.shape[0]).cuda())    

    pruned_ratio = pruned/total

    print(f'PRUNE RATIO={pruned_ratio}')
    print('PREPROCESSING SUCCESSFUL!')

    return cfg_mask, cfg

def _copy_weight(cfg_mask, model, newmodel):

    layer_id_in_cfg = 0
    #TODO: start_mask = torch.ones(3), 3為input channel(R,G,B)
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]
    start_mask_downsample = start_mask.clone()
    end_mask_downsample = end_mask.clone()

    for [m0, m1, (m1_name, _)] in zip(model.modules(), newmodel.modules(), newmodel.named_modules()):

        # 檢查權重內容
        # print('====================================================')
        # print('m1_name:                    ', m1_name)
        # print('m1:                         ', m1)
        # print('layer_id_in_cfg:            ', layer_id_in_cfg)
        # print('start mask shape:           ', start_mask.shape)
        # print('end mask shape:             ', end_mask.shape)
        # print('start_mask_downsample shape:', start_mask_downsample.shape)
        # print('end_mask_downsample shape:  ', end_mask_downsample.shape)

        if isinstance(m0, nn.BatchNorm2d):

            # 處理剪枝後的權重
            m0.weight.data.mul_(end_mask)
            m0.bias.data.mul_(end_mask)

            # 找出遮罩中非零元素的 index
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 從 mask 中將值為 1 的 positions 全部找出來變成一個 array
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            # 將原本模型的權重複製到剪枝模型的權重
            # 複製 weight 與 bias
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()

            # 複製 running mean 跟 running variance
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()

            # 最後一層連接層不做修改
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            #! 這裡的 start_mask_downsample 跟 end_mask_downsample 是為了處理 downsample 時 channel 對應的部分
            if 'conv1' in m1_name:
                start_mask_downsample = start_mask.clone()
            if 'conv3' in m1_name:
                end_mask_downsample = end_mask.clone()

            # 將原本模型的捲積層權重複製到對應剪枝模型卷積層的權重
            #! 當 downsample 時， channel 的 mask 改用 start_mask_downsample 跟 end_mask_downsample
            if 'downsample.0' in m1_name:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask_downsample.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask_downsample.cpu().numpy())))
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

            #! 避免 channel 數量為 1 時被 squeeze 掉，所以要 resize
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            w = m0.weight.data[:, idx0, :, :].clone()
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()

            #! 因為 conv3 後沒有 bn (start_mask 和 end_mask 的更新在 bn)，過完 conv3 要先更新 start_mask 跟 end_mask
            if 'conv3' in m1_name:
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]


        elif isinstance(m0, nn.Linear):
            # 參考 https://pytorch.org/docs/stable/generated/torch.nn.Linear.html 來決定該如何複製Linear Layer參數
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))

            # 複製 weight
            m1.weight.data = m0.weight.data[:, idx0].clone() # [out_channel, in_channel]

            # 複製 bias
            m1.bias.data = m0.bias.data.clone()

    return newmodel

def prune(opt, model):
    # get channel mask
    cfg_mask, cfg = _get_channel_mask(opt.PRUNE_PERCENT)

    # create new model with cfg
    newmodel =  resnet50(cfg).to(opt.device)

    # copy weight
    newmodel = _copy_weight(cfg_mask, model, newmodel)

    return newmodel, cfg

def save_model(model, cfg, path):
    torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet50 Pruning')
    opt = arg(parser)

    # load model
    model = resnet50().to(opt.device)
    model.load_state_dict(torch.load(opt.PATH, map_location=torch.device('cpu')))

    # create pruned model
    newmodel, cfg = prune(opt, model)
    # torchsummary.summary(newmodel, (1, 28, 28))
    # print(newmodel)
    
    # save pruned model
    save_model(newmodel, cfg, opt.PRUNE_PATH)

    # create dataloader
    train_loader, test_loader = create_dataloader(opt)

    # test pruned model
    test(opt, newmodel, test_loader)