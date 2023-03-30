import os
import sys
from glob import glob
import numpy as np
import torch
from torch import no_grad,load,cuda
from monai.data import Dataset,DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.networks.layers import Norm, Act
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    LoadImaged,
    SaveImage,
    Resized,
    EnsureTyped,
    EnsureChannelFirstd,
    EnsureType,
    )
import SimpleITK as sitk
import shutil
import xlsxwriter
from natsort import natsorted,ns
from PyQt5.QtCore import *

class Pred(QThread):
    _signal1 = pyqtSignal(int)
    def __init__(self):
        super(Pred,self).__init__()
        self.num_class = 2
        self.pred_data=np.array([])
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.makedirs('temp')


    def load_image(self,image_path):
        self.image_dir = image_path


    def load_model(self,modelpath):
        self.modelpath = modelpath

    def _normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def dicom2npy(self):
        dcms_path=self.image_dir
        filename = os.path.basename(dcms_path)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
        reader.SetFileNames(dicom_names)
        image2 = reader.Execute()
        # 2.将整合后的数据转为array，并获取dicom文件基本信息
        image_array = sitk.GetArrayFromImage(image2)  # z, y, x
        image_array = self._normalization(image_array)
        for i in range(image_array.shape[0]):
            name = filename+f'_{i}.npy'
            savename = 'temp/' + name
            np.save(savename, image_array[i])


    def dataload(self):
        self.dicom2npy()
        test_images = natsorted(
            glob(os.path.join('temp', '*.npy')), alg=ns.PATH)
        test_data = [{"image": image} for image in test_images]

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Resized(keys=["image"], spatial_size=(512, 512),mode=("bilinear")),
                EnsureTyped(keys=["image"]),
            ]
        )
        test_ds = Dataset(data=test_data, transform=test_transforms)  # val_files
        self.test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)



    def run(self):
        self.dataload()
        device='cuda' if cuda.is_available() else 'cpu'
        post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        model = SegResNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2
        ).to(device)
        model.load_state_dict(load(self.modelpath,map_location=device))
        model.eval()
        with no_grad():
            for index, test_data in enumerate(self.test_loader):
                test_images= test_data["image"].to(device)

                test_outputs = model(test_images)
                test_outputs = [post_trans(i) for i in test_outputs]

                meta_data = decollate_batch(test_data["image_meta_dict"])

                for test_output, data in zip(test_outputs, meta_data):
                    test_output = test_output[1, :, :].unsqueeze(0)

                    pred_slice_data = (torch.tensor([item.cpu().detach().numpy()
                                               for item in test_output]).cuda()).to('cpu').numpy()
                if self.pred_data.size==0:
                    self.pred_data = pred_slice_data
                else:
                    self.pred_data = np.append(self.pred_data, pred_slice_data, axis=0)
            self._signal1.emit(1)

    def get_pred(self):
        return self.pred_data
        # else:
        #     pass







# if __name__ == '__main__':
#     a=Pred()
#     a.load_image('D:\MyProject\Synovinm\case')
#     a.load_model('model/model.pth')
#     a.load_savepath('result')
#     a.run()

