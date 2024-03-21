import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import logging
from tool.utils import *
from tool.logger import get_root_logger
from model.EncoderDecoder import EncoderDecoder
from torch.utils.data import DataLoader
from tool.istddata import CustomDataset
from model.SwinTransformer import SwinTransformer

def main(seed):
    cfg = load_config("istd.yaml")
    mkdir_or_exist(cfg['ckpt_model'])
    mkdir_or_exist(cfg['ckpt_image'])
    logger = get_root_logger(name='swin', log_file=cfg['logfile'], log_level=logging.INFO)
    logger.info(seed)


    '''  2. datasets  '''
    image_size = cfg['swin_b_p4_w12_384']['img_size']
    train_dataset = CustomDataset(image_size)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)


    '''  3. initial  '''
    logger.info("build model...")
    swin = SwinTransformer(**cfg['swin_b_p4_w12_384'])
    model = EncoderDecoder(swin)
    checkpoint = torch.load("./model/swin_base_patch4_window12_384_22k.pth", map_location='cpu')
    model.backbone.load_state_dict(checkpoint['model'], strict=False)
    model.cuda().train()
    # logger.info(model)
    # logger.info("model parameters: {}".format(sum(param.numel() for param in model.parameters())/1e6))


    '''  4. loss && optimizer'''
    if cfg['loss']=='MyWcploss':
        bce = MyWcploss().cuda()

    if cfg['optimizer']['use']=='SGD':
        optimizer = torch.optim.SGD([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3, \
                'weight_decay': cfg['optimizer']['weight_decay']}], momentum=cfg['optimizer']['momentum']
            )
    elif cfg['optimizer']['use']=='Adam':
        optimizer = torch.optim.Adam([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3}], \
                betas=(0.9, 0.999), eps=1e-8
            )
    elif cfg['optimizer']['use']=='AdamW':
        optimizer = torch.optim.AdamW([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3}], \
                betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg['optimizer']['weight_decay'])


    '''  5. train  '''
    step=0
    train_loader = iter(train_loader)
    logger.info("start training")
    model.train()
    while step<=cfg['iter']:
        samples = next(train_loader)
        optimizer.param_groups[0]['lr'] = 2*cfg['optimizer']['lr'] * (1 - float(step) / cfg['iter']) ** 0.9
        optimizer.param_groups[1]['lr'] = cfg['optimizer']['lr'] * (1 - float(step) / cfg['iter']) ** 0.9
        # train one iteration
        optimizer.zero_grad()
        img, lab = samples['O'], samples['B']
        img, lab = img.cuda(), lab.cuda()
        attn, shad4, shad3, shad2, shad1, shadlf, fuse_pred = model(img)

        lossa = bce(attn, lab)
        loss4 = bce(shad4, lab)
        loss3 = bce(shad3, lab)
        loss2 = bce(shad2, lab)
        loss1 = bce(shad1, lab)
        lossf = bce(shadlf, lab)
        lossp = bce(fuse_pred, lab)

        loss = lossa+loss4+loss3+loss2+loss1+lossf+lossp
        predicts = torch.sigmoid(fuse_pred)

        if step%20==0:
            logger.info("step: %d loss all: %.2f" % (step, loss.item()))

        loss.backward()
        optimizer.step()

        if step%50==0:
            save_mask(cfg['ckpt_image'], step, image_size, [samples['image'], samples['B'], (samples['B'], predicts)])
            logger.info("log: %d" % (step))
        if step%500==0:
            save_ckpts(cfg['ckpt_model'], model, step, optimizer, "latest.pth")
        if step%2000==0:
            save_ckpts(cfg['ckpt_model'], model, step, optimizer, str(step)+'.pth')
        step+=1


if __name__ == '__main__':
    seed = np.random.randint(10000000)
    init_seeds(seed)
    main(seed)