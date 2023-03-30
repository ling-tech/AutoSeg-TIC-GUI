from PyQt5.QtWidgets import QFileDialog
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np

path = "/home/linv/桌面/121661(2017.12.1).r"
os.makedirs(path+'/'+'img')
