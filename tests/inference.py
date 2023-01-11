import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np
import glob
import time
import pickle
import torchvision

from torchvision.io import read_image
from torchvision import transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt
from medusa_covnet import CovNet


model_file = sys.argv[1]  # input model pickle file
input_dir = sys.argv[2]  # input image dir for inference

def main():
    print(torchvision.__version__)
    batch_size = 40

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(batch_size),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CovNet()
    model.to(device)
    
    for im in glob.glob(os.path.join(input_dir, "*.png")):
        print(im)
        image = read_image(im)

        img = image.to(device)

        transformed = transform(img[:3]).to(device)
        # timg = T.ToPILImage()(transformed.to('cpu'))
        # plt.imshow(np.asarray(timg))
        # plt.show()

        # # model.load_state_dict(torch.load("./medusa_CNN0.pt"))
        model = torch.load(model_file)
        model.eval()
        #
        with torch.no_grad():
            output = model.forward(transformed[None, ...])  # insert singleton batch since model expects batch input
        scores, prediction = torch.max(output.data, 1)

        # output = model(transformed[None, ...])
        print(prediction)
        print('\n')


if '__main__':
    main()
