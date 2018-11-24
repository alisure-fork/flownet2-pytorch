import os
import cv2
import torch
import models
import argparse
import numpy as np
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


class MyDataVideo(data.Dataset):

    def __init__(self, cap, skip=6, is_video=True):
        self.cap = cap
        self.skip = skip + 2
        self.is_video = is_video
        pass

    def __getitem__(self, index):
        if self.cap.isOpened():
            ret1, img1 = self.cap.read()
            if not ret1:
                raise Exception(".............")

            [self.cap.read() for i in range(self.skip - 2)]

            ret2, img2 = self.cap.read()
            if not ret2:
                raise Exception(".............")

            frame_size = img1.shape[0:2]
            render_size = [((frame_size[0]) // 64) * 64, ((frame_size[1]) // 64) * 64]

            cropper = StaticCenterCrop(frame_size, render_size)
            images_ori = list(map(cropper, [img1, img2]))
            images = np.array(images_ori).transpose(3, 0, 1, 2)
            images = torch.from_numpy(images.astype(np.float32))

            return images, images_ori
        else:
            raise Exception(".............")
            pass

    def __len__(self):
        return int(self.cap.get(7)) // self.skip if self.is_video else 1000000

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


def flow2img(is_file, flow_path_or_file='./work2/run', png_path_or_file='./work2/run-png'):
    if is_file:
        ml = './read/C/color_flow ' + flow_path_or_file + " " + png_path_or_file
        os.system(ml)
    else:
        flo_files = os.listdir(flow_path_or_file)

        if not os.path.exists(png_path_or_file):
            os.makedirs(png_path_or_file)
            pass

        for flo_file_index, flo_file in enumerate(flo_files):
            ml = './read/C/color_flow ' + os.path.join(flow_path_or_file, flo_file) + " " + os.path.join(
                png_path_or_file, "{}.png".format(os.path.splitext(flo_file)[0]))
            os.system(ml)
            pass
        pass
    pass


if __name__ == '__main__':
    args = arguments()

    number_gpu = torch.cuda.device_count()

    # 数据集
    video_file = '/home/ubuntu/data1.5TB/异常dataset/ShanghaiTech/train/02_001.avi'
    # video_file = 0
    inference_data_set = MyDataVideo(cap=cv2.VideoCapture(video_file), skip=4, is_video=False)
    inference_loader = DataLoader(inference_data_set, 2, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    # model
    inference_model = InferenceModel(args, models.FlowNet2).cuda()
    inference_model = nn.parallel.DataParallel(inference_model, device_ids=list(range(number_gpu)))

    # 加载模型
    checkpoint_path = './model/FlowNet2_checkpoint.pth.tar'
    load_model(checkpoint_path)

    # 新建文件夹
    flow_folder = "{}/{}".format("./work2", "run")
    png_folder = "{}/{}".format("./work2", "run-png")
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder)
        os.makedirs(png_folder)
    else:
        os.system("rm -r {}".format(flow_folder))
        os.makedirs(flow_folder)
        os.system("rm -r {}".format(png_folder))
        os.makedirs(png_folder)
        pass

    # Reusable function for inference
    def inference(data_loader, model):
        model.eval()

        progress = tqdm(data_loader, ncols=100, total=len(data_loader), desc='Inferencing ', leave=True, position=0)
        for batch_idx, (_data, _data_ori) in enumerate(progress):
            _data = Variable(_data.cuda(async=True))

            with torch.no_grad():
                output = model(_data)
                pass

            bz = output.shape[0]
            for i in range(bz):
                image_flow = os.path.join(flow_folder, '%06d.flo' % (batch_idx * bz + i))
                image_png = os.path.join(png_folder, '%06d.png' % (batch_idx * bz + i))

                _flow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow(image_flow, _flow)

                flow2img(is_file=True, flow_path_or_file=image_flow, png_path_or_file=image_png)

                img = cv2.imread(image_png)
                img_ori = _data_ori[i].data.cpu().numpy()[0]
                result = np.concatenate((img_ori, img), axis=0)

                cv2.imshow('double', result)
                if cv2.waitKey(30) & 0xff == "q":
                    break
                pass

            progress.update(1)
            pass
        progress.close()
        pass

    inference(data_loader=inference_loader, model=inference_model)

    inference_data_set.cap.release()
    cv2.destroyAllWindows()

    pass
