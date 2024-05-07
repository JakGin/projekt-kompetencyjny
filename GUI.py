import sys
from PIL import Image
import io
from PySide6 import QtCore, QtWidgets, QtGui


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        self.save_path = "./image"
        self.img = None
        super().__init__()
        self.filename = ""

        self.inputButton = QtWidgets.QPushButton("")
        self.analyzeButton = QtWidgets.QPushButton("Analyze image.")
        self.text = QtWidgets.QLabel("Input image to analyze.", alignment=QtCore.Qt.AlignCenter)
        self.pixmap = QtGui.QPixmap()

        self.inputButton.setFixedSize(500, 500)
        self.inputButton.setIconSize(QtCore.QSize(400, 400))
        self.inputButton.setIcon(QtGui.QIcon("./jpg.png"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.inputButton, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.analyzeButton)

        self.inputButton.clicked.connect(self.browse_files)
        self.analyzeButton.clicked.connect(self.analyze_image)

    @QtCore.Slot()
    def analyze_image(self):
        self.text.setText("Analyzing image.")

    def browse_files(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select Image", dir="C:\\Users\\User\\Desktop\\")
        self.filename = filename[0]
        self.save_image()

    def save_image(self):
        self.text.setText("Image inputted.")
        allowed_format = ["png", "jpg", "jpeg"]
        if self.filename.split('/')[-1].split('.')[-1] not in allowed_format:
            return None
        self.img = Image.open(self.filename)
        temp_img = self.img.copy()
        temp_img = temp_img.resize((400, 400))
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format=allowed_format[0])
        img_byte_arr.seek(0)
        self.pixmap.loadFromData(img_byte_arr.read())
        self.inputButton.setIcon(QtGui.QIcon(self.pixmap))
        self.img.save(self.save_path + "/" + self.filename.split('/')[-1])


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.setStyleSheet('font-size: 30px')
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
