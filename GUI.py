import sys
from PIL import Image
import io
from PySide6 import QtCore, QtWidgets, QtGui
#import cnn_to_app as cnn

# kolorystyka Aplle (Mac)


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, text: str = "Test"):
        super().__init__()

        self.setWindowTitle("HELLO!")
        self.resize(150, 100)
        self.setStyleSheet("font-size: 25px")
        QBtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel(text)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyze architecture style")
        self.save_path = "./image"
        self.img = None
        self.filename = ""
        self.setStyleSheet("""
                    QWidget {
                        background-color: #FFFFFF; /* Light grey */
                        padding: 15px; /* Padding inside the widget */
                    }
                    QLabel {
                        color: black; /* Dark grey text color */
                        font-size: 30px;
                    }
                """)
        self.inputButton = QtWidgets.QPushButton("")
        self.analyzeButton = QtWidgets.QPushButton("Analyze image.")
        self.analyzeButton.setStyleSheet("""
        QPushButton {
                        appearance: none;
  backface-visibility: hidden;
  background-color: #038DC8;
  border-radius: 10px;
  border-style: none;
  box-shadow: none;
  box-sizing: border-box;
  color: #fff;
  cursor: pointer;
  display: inline-block;
  font-family: Inter,-apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;
  font-size: 30px;
  font-weight: 500;
  height: 50px;
  letter-spacing: normal;
  line-height: 1.5;
  outline: none;
  overflow: hidden;
  padding: 14px 30px;
  position: relative;
  text-align: center;
  text-decoration: none;
  transform: translate3d(0, 0, 0);
  transition: all .3s;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: top;
  white-space: nowrap;
                    }
            QPushButton:hover {
  background-color: #69C7F1;
  box-shadow: rgba(0, 0, 0, .05) 0 5px 30px, rgba(0, 0, 0, .05) 0 1px 4px;
  opacity: 1;
  transform: translateY(0);
  transition-duration: .35s;
}
QPushButton:active {
  box-shadow: rgba(0, 0, 0, .1) 0 3px 6px 0, rgba(0, 0, 0, .1) 0 0 10px 0, rgba(0, 0, 0, .1) 0 1px 4px -1px;
  transform: translateY(2px);
  transition-duration: .35s;
}
QPushButton:disabled {
background-color: #909090;
}
        """)
        self.analyzeButton.setDisabled(True)
        self.text = QtWidgets.QLabel("Input image to analyze.", alignment=QtCore.Qt.AlignCenter)
        self.pixmap = QtGui.QPixmap()

        self.inputButton.setFixedSize(500, 500)
        self.inputButton.setIconSize(QtCore.QSize(400, 400))
        self.inputButton.setStyleSheet(("""
            QPushButton {
                        appearance: none;
  backface-visibility: hidden;
  background-color: #038DC8;
  border-radius: 10px;
  border-style: none;
  box-shadow: none;
  box-sizing: border-box;
  color: #fff;
  cursor: pointer;
  display: inline-block;
  font-family: Inter,-apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;
  font-size: 30px;
  font-weight: 500;
  height: 50px;
  letter-spacing: normal;
  line-height: 1.5;
  outline: none;
  overflow: hidden;
  padding: 14px 30px;
  position: relative;
  text-align: center;
  text-decoration: none;
  transform: translate3d(0, 0, 0);
  transition: all .3s;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: top;
  white-space: nowrap;
                    }
            QPushButton:hover {
  background-color: #69C7F1;
  box-shadow: rgba(0, 0, 0, .05) 0 5px 30px, rgba(0, 0, 0, .05) 0 1px 4px;
  opacity: 1;
  transform: translateY(0);
  transition-duration: .35s;
}
QPushButton:active {
  box-shadow: rgba(0, 0, 0, .1) 0 3px 6px 0, rgba(0, 0, 0, .1) 0 0 10px 0, rgba(0, 0, 0, .1) 0 1px 4px -1px;
  transform: translateY(2px);
  transition-duration: .35s;
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
        #dlg = CustomDialog()
        #if dlg.exec():
        #    print("Success!")
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
        self.analyzeButton.setDisabled(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.setStyleSheet('font-size: 30px')
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
