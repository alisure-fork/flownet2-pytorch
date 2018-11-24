import os
import torch
import models
import argparse
import numpy as np
from glob import glob
import torch.nn as nn
from tqdm import tqdm
from utils import flow_utils
import torch.utils.data as data
from torch.autograd import Variable
from datasets import StaticCenterCrop
from torch.utils.data import DataLoader
import utils.frame_utils as frame_utils


class InferenceModel(nn.Module):

    def __init__(self, args, model):
        super(InferenceModel, self).__init__()
        self.model = model(args)
        pass

    def forward(self, data):
        return self.model(data)

    pass


class MyData(data.Dataset):

    def __init__(self, image_root='', dstype='clean', replicates=1):
        self.render_size = [-1, -1]
        self.replicates = replicates

        flow_root = os.path.join(image_root, 'flow')
        image_root = os.path.join(image_root, dstype)

        file_list = sorted(glob(os.path.join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root) + 1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = os.path.join(image_root, fprefix + "%04d" % (fnum + 0) + '.png')
            img2 = os.path.join(image_root, fprefix + "%04d" % (fnum + 1) + '.png')

            if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        images = [img1, img2]
        image_size = img1.shape[:2]

        cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))

        return [images]

    def __len__(self):
        return self.size * self.replicates

    pass


class MyData2(data.Dataset):

    def __init__(self, image_root):
        self.image_list = [os.path.join(image_root, image) for image in os.listdir(image_root)]
        self.frame_len = len(self.image_list) - 1
        self.frame_size = frame_utils.read_gen(self.image_list[0]).shape

        self.render_size = [((self.frame_size[0]) // 64) * 64, ((self.frame_size[1]) // 64) * 64]
        pass

    def __getitem__(self, index):
        index = index % self.frame_len

        img1 = frame_utils.read_gen(self.image_list[index])
        img2 = frame_utils.read_gen(self.image_list[index + 1])
        image_size = img1.shape[:2]

        cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, [img1, img2]))
        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))
        return [images]

    def __len__(self):
        return self.frame_len

    pass


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    return parser.parse_args()


def load_model(_checkpoint_path):
    if _checkpoint_path and os.path.isfile(_checkpoint_path):
        print("Loading checkpoint '{}'".format(_checkpoint_path))
        checkpoint = torch.load(_checkpoint_path)
        inference_model.module.model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint found at '{}'".format(_checkpoint_path))
        quit()
        print("Random initialization")
        pass
    pass

if __name__ == '__main__':
    args = arguments()

    number_gpus = torch.cuda.device_count()

    # 数据集
    inference_data = '../data/MPI-Sintel/training'
    inference_dataset = MyData(image_root=inference_data)
    inference_loader = DataLoader(inference_dataset, 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # model
    inference_model = InferenceModel(args, models.FlowNet2C).cuda()
    inference_model = nn.parallel.DataParallel(inference_model, device_ids=list(range(number_gpus)))

    # 加载模型
    checkpoint_path = './model/FlowNet2-C_checkpoint.pth.tar'
    load_model(checkpoint_path)

    flow_folder = "{}/{}".format("./work2", "run")
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder)

    # Reusable function for inference
    def inference(data_loader, model):
        model.eval()

        progress = tqdm(data_loader, ncols=100, total=len(data_loader), desc='Inferencing ', leave=True, position=0)
        for batch_idx, data in enumerate(progress):
            data = [Variable(d.cuda(async=True)) for d in data]

            with torch.no_grad():
                output = model(data[0])
                pass

            _flow = output[0].data.cpu().numpy().transpose(1, 2, 0)
            flow_utils.writeFlow(os.path.join(flow_folder, '%06d.flo' % batch_idx), _flow)

            progress.update(1)
            pass
        progress.close()
        pass

    inference(data_loader=inference_loader, model=inference_model)

    pass
