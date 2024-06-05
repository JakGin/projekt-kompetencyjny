import sys
from PIL import Image
import io
from PySide6 import QtCore, QtWidgets, QtGui
#import cnn_to_app as cnn


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.save_path = "./image"
        self.img = None
        self.filename = ""
        self.setStyleSheet("""
                    QWidget {
                        background-color: #BDBCBF; /* Light grey */
                        padding: 15px; /* Padding inside the widget */
                    }
                    QLabel {
                        color: black; /* Dark grey text color */
                        font-size: 30px;
                    }
                    QPushButton {
                        align-items: center;
  appearance: none;
  background-color: #DEDEDE;
  border-radius: 24px;
  border-style: none;
  box-shadow: rgba(0, 0, 0, .2) 0 3px 5px -1px,rgba(0, 0, 0, .14) 0 6px 10px 0,rgba(0, 0, 0, .12) 0 1px 18px 0;
  box-sizing: border-box;
  color: #3c4043;
  cursor: pointer;
  display: inline-flex;
  fill: currentcolor;
  font-family: "Google Sans",Roboto,Arial,sans-serif;
  font-size: 14px;
  font-weight: 500;
  justify-content: center;
  letter-spacing: .25px;
  line-height: normal;
  overflow: visible;
  padding: 2px 24px;
  position: relative;
  text-align: center;
  text-transform: none;
  transition: box-shadow 280ms cubic-bezier(.4, 0, .2, 1),opacity 15ms linear 30ms,transform 270ms cubic-bezier(0, 0, .2, 1) 0ms;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  will-change: transform,opacity;
  z-index: 0;
                    }
                    QPushButton:hover {
                        background: #F6F9FE;
  color: #174ea6;
                    }
                QPushButton:active {
                box-shadow: 0 4px 4px 0 rgb(60 64 67 / 30%), 0 8px 12px 6px rgb(60 64 67 / 15%);
  outline: none;
                }
                """)
        self.inputButton = QtWidgets.QPushButton("")
        self.analyzeButton = QtWidgets.QPushButton("Analyze image.")
        self.analyzeButton.setStyleSheet("""
        QPushButton {
            position: absolute;
            width: 750px;
            height: 70px;
        }
        """)
        self.text = QtWidgets.QLabel("Input image to analyze.", alignment=QtCore.Qt.AlignCenter)
        self.pixmap = QtGui.QPixmap()

        self.inputButton.setFixedSize(500, 500)
        self.inputButton.setIconSize(QtCore.QSize(400, 400))
        self.inputButton.setStyleSheet(("""
            QPushButton {
                position: absolute;
                width: 500px;
                height: 500px;
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
        #class_name, score = cnn.classify_image(self.filename)
        #self.text.setText("{}".format(class_name))

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
