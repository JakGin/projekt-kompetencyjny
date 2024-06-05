import sys
from PIL import Image
import io
from PySide6 import QtCore, QtWidgets, QtGui
import cnn_to_app as cnn


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.save_path = "./image"
        self.img = None
        self.filename = ""
        self.setStyleSheet("""
                    QWidget#myCustomWidget {
                        background-color: #f0f0f0; /* Light grey */
                        padding: 15px; /* Padding inside the widget */
                    }
                    QLabel {
                        color: black; /* Dark grey text color */
                        font-size: 30px;
                    }
                    QPushButton {
                        background-color: #d3d3d3; 
    border: none; 
    border-radius: 15px; 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
    font-size: 20px; 
    color: #000; /* Text color */
    text-align: center; /* Center the text */
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer; /* Pointer cursor on hover */
    transition: background-color 0.3s ease; 
                        text-align: center;
                        font-size: 30px;
                    }
                    QPushButton:hover {
                        background-color: #c0c0c0;
                        color: black;
                        border: 2px solid #171717;
                    }
                """)
        self.inputButton = QtWidgets.QPushButton("")
        self.analyzeButton = QtWidgets.QPushButton("Analyze image.")
        self.analyzeButton.setStyleSheet("""
        QPushButton {
            position: absolute;
            width: 745px;
            height: 66.5px;
        }
        """)
        self.text = QtWidgets.QLabel("Input image to analyze.", alignment=QtCore.Qt.AlignCenter)
        self.pixmap = QtGui.QPixmap()

        self.inputButton.setFixedSize(500, 500)
        self.inputButton.setIconSize(QtCore.QSize(400, 400))
        self.inputButton.setStyleSheet(("""
            QPushButton {
                position: absolute;
                width: 495px;
                height: 495px;
            }
        """))
        self.inputButton.setIcon(QtGui.QIcon("./image.png"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.inputButton, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.analyzeButton)

        self.inputButton.clicked.connect(self.browse_files)
        self.analyzeButton.clicked.connect(self.analyze_image)

    @QtCore.Slot()
    def analyze_image(self):
        self.text.setText("Analyzing image.")
        class_name, score = cnn.classify_image(self.filename)
        self.text.setText("{}".format(class_name))

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


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.setStyleSheet('font-size: 30px')
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
