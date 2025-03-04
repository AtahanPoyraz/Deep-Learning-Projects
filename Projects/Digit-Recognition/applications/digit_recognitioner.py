from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageOps
import numpy as np
import sys

from keras.api.models import load_model

class DigitRecognitioner(QtWidgets.QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.model = load_model(path)

        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(300, 300)
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Prediction: ')
        self.prediction.setFont(QtGui.QFont('Monospace', 11))

        self.button_clear = QtWidgets.QPushButton('CLEAR')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_save = QtWidgets.QPushButton('PREDICT')
        self.button_save.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.container)
        self.setCentralWidget(central_widget)

    def clear_canvas(self):
        self.prediction.setText(f'Prediction: ')
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.update()

    def predict(self):
        arr = np.frombuffer(self.label.pixmap().toImage().bits().asarray(300 * 300 * 4), dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS)))
        arr = (arr / 255.0).reshape(28, 28, 1) 
        arr = np.expand_dims(arr, axis=0)  

        prediction = self.model.predict(arr)  
        predicted_label = np.argmax(prediction, axis=1)[0]

        self.prediction.setText(f'Prediction: {predicted_label}')
        
    def mouseMoveEvent(self, e):
        if self.last_x is None:  
            self.last_x = e.x()
            self.last_y = e.y()
            return  

        painter = QtGui.QPainter(self.label.pixmap())
        p = painter.pen()
        p.setWidth(12)
        self.pen_color = QtGui.QColor('#FFFFFF')
        p.setColor(self.pen_color)
        painter.setPen(p)

        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = DigitRecognitioner("../artifacts/model.keras")
    window.setWindowTitle('Digit Predictor')
    window.resize(300, 300)
    window.show()

    sys.exit(app.exec_())
