import sys
import numpy as np

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.input_label = QLabel("Введите данные для графика:")
        self.input_line_edit = QLineEdit()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_line_edit)
        layout.addWidget(self.button)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot(self):
        data = [random.random() for i in range(10)]
        formula = self.input_line_edit.text()

        x_ = np.linspace(0, 100, 101)
        y_ = [eval(formula, {'__builtins__': None}, {'x': i}) for i in x_]

        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.plot(x_, y_)
        plt.grid()

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
