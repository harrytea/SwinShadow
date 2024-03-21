import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import argparse
import numpy as np
from tool.utils import *
from tqdm import tqdm
from PIL import Image
from tool.sbudata import TestDataset
from tool.misc import crf_refine
from model.EncoderDecoder import EncoderDecoder
from model.SwinTransformer import SwinTransformer

# SBU
sbu_image = r"/data/wangyh/data4/Datasets/shadow/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages"
sbu_mask  = r"/data/wangyh/data4/Datasets/shadow/SBU-shadow/SBU-shadow/SBU-Test/ShadowMasks"
# ISTD
istd_image = r"/data/wangyh/data4/Datasets/shadow/ISTD_Dataset/test/test_A"
istd_mask  = r"/data/wangyh/data4/Datasets/shadow/ISTD_Dataset/test/test_B"
# UCF
ucf_image = '/data/wangyh/data4/Datasets/shadow/UCF-shadow/InputImages'
ucf_mask = '/data/wangyh/data4/Datasets/shadow/UCF-shadow/GroundTruth'


parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',   default='./sbu/ckpt', help='directory for checkpoints')
parser.add_argument('--save_dir',   default='./results/',  help='directory for checkpoints')
parser.add_argument('--batch_size', default=1, type=int,     help='number of samples in one batch')
parser.add_argument('--image_size',   default=384,  help='directory for checkpoints')
parser.add_argument('--swin_model',  default='swin_b_p4_w12_384', help='chose vit model')
args = parser.parse_args()



def main():
    for ee in range(5500, 5501, 5000):
        args.save_dir ='./results/' +"sbu/" +str(ee)+"lowft"
        cfg = load_config("sbu.yaml")
        print(args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        test_dataset    = TestDataset(args.image_size)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=False)

        print("build model...")
        swin = SwinTransformer(**cfg[args.swin_model])
        model = EncoderDecoder(swin)
        load_checkpoints(model, args.ckpt_dir, str(ee)+".pth")
        model.cuda().eval()

        for i, (batch, file_path) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                O, B,= batch['O'], batch['B']
                O, B = O.cuda(), B.cuda()
                _, shad4, shad3, shad2, shad1, shadlf, fuse_pred = model(O)

                predict = torch.sigmoid(shadlf)
                image = Image.open(os.path.join(sbu_image, file_path[0])).convert('RGB')
                final = Image.fromarray((predict.cpu().data * 255).numpy().astype('uint8')[0,0,:,:])
                final = np.array(final.resize(image.size))
                final_crf = crf_refine(np.array(image), final)
                io.imsave(os.path.join(args.save_dir, file_path[0]), final_crf)


if __name__ == '__main__':
    main()
