import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PySide6.QtCore import *
from PySide6.QtWidgets import *


class MPLGraph(FigureCanvasQTAgg):
    def __init__(self, z, t, a_z_t):
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(1, 2), layout="tight")
        self.ax1.grid()
        self.ax2.grid()

        self.a = a_z_t
        self.t, self.z = t, z
        self.zi, self.ti = 0, 0
        self.a_t, self.a_z = self.a[self.zi, :], self.a[:, self.ti]
        self.line_t, = self.ax1.plot(self.t, self.compute_a_t())
        self.line_z, = self.ax2.plot(self.z, self.compute_a_z())
        self.ax1.set_xlim([0, self.t[-1]])
        self.ax2.set_xlim([0, self.z[-1]])
        self.ax1.set_ylim([self.a.min(), self.a.max()])
        self.ax2.set_ylim([self.a.min(), self.a.max()])
        self.ax1.set_title(f'A({self.z[self.zi]}, t)')
        self.ax2.set_title(f'A(z, {self.t[self.ti]})')
        super().__init__(fig)

    def compute_a_t(self):
        return self.a[:, self.zi]

    def compute_a_z(self):
        return self.a[self.ti, :]

    def update_a_z(self):
        self.line_z.set_ydata(self.compute_a_z())
        self.ax2.set_title(f'A(z, {self.t[self.ti]})')
        self.draw()

    def update_a_t(self):
        self.line_t.set_ydata(self.compute_a_t())
        self.ax1.set_title(f'A({self.z[self.zi]}, t)')
        self.draw()

    def update_t(self, slider_t):
        self.ti = int(len(self.t) * slider_t / 10001)
        self.update_a_z()

    def update_z(self, slider_z):
        self.zi = int(len(self.z) * slider_z / 10001)
        self.update_a_t()


class HeightSlider(QSlider):
    def __init__(self):
        super().__init__(Qt.Vertical)
        self.setMinimum(0)
        self.setMaximum(10000)
        self.setValue(0)


class MainWindow(QMainWindow):
    def __init__(self, z, t, a_z_t):
        super().__init__()
        self.setGeometry(300, 100, 800, 600)
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # widgets
        graph = MPLGraph(z, t, a_z_t)
        slider_z = HeightSlider()
        slider_t = HeightSlider()

        # layout
        layout.addWidget(slider_z, 1, 1)
        layout.addWidget(slider_t, 2, 1)
        layout.addWidget(graph, 1, 3, 2, 1)

        # connections
        slider_z.valueChanged.connect(graph.update_z)
        slider_t.valueChanged.connect(graph.update_t)