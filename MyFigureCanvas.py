# −∗− coding: utf−8 −∗−
""" MATPLOTLIB WIDGET """
# Python Qt5 bindings for GUI objects
from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout
# import the Qt5Agg FigureCanvas object , that binds Figure to
# Qt5Agg backend. It also inherits from QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Matplotlib Toolbar
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
# Matplotlib Figure object
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams['font.size'] = 9
import SimpleITK as sitk

class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""

    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        # initialization of the canvas
        FigureCanvas.__init__(self, self.fig)
        # we define the widget as expandable
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)


class ImageReader():
    def __init__(self,path,mode):
        self.path = path
        self.mode = mode
        self._get_img()

    def _get_img(self):
        if self.mode == 'NIFTI':
            self.img = sitk.ReadImage(self.path)
            # self.img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        elif self.mode == 'DICOM':
            reader = sitk.ImageSeriesReader()
            img_names = reader.GetGDCMSeriesFileNames(self.path)
            reader.SetFileNames(img_names)
            self.img = reader.Execute()
            # self.img.SetDirection([1,0,0,0,1,0,0,0,1])

    def get_array(self):
        img_fdata = sitk.GetArrayFromImage(self.img)
        return img_fdata

    def get_spacing(self):
        spacing = self.img.GetSpacing()
        return spacing