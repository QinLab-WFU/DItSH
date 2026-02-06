import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from networks.ca_net import CANet
import json
from cam import GradCAM, show_cam_on_image, center_crop_img
import random


def main():
    # 由于每次生成的灰度热力图都不一样，所以使用下面几行去除随机性
    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.alexnet(pretrained=True)
    # target_layers = [model.features]
    model = CANet(32,17,"resnet50").cuda()

    # model = cnn_model.CNNNet("alexnet", 64, 30)
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    weights_path = "/home/admin01/桌面/CAN/para_model/UCMD32/best.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"),strict=False)
    # model.eval()
    target_layers = [model.backbone.conv_block3[2]]
    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/home/admin01/桌面/CAN/airplane03.tif"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    input_tensor = input_tensor.to(device='cuda')

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = None  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()


