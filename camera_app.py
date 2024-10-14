import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Luo layout ja label kuvien näyttämiseen
        self.layout = QHBoxLayout()
        self.camera_label = QLabel(self)
        self.layout.addWidget(self.camera_label)
        self.setLayout(self.layout)

        # Tekstilabel, johon voi lisätä tekstiä oikealle
        self.text_label = QLabel("Tämä on tekstilaatikko", self)
        self.layout.addWidget(self.text_label)

        self.setLayout(self.layout)
        
        # Avaa kamera (0 tarkoittaa oletuskameraa)
        self.cap = cv2.VideoCapture(0)
        
        # Luo QTimer, joka päivittyy 30ms välein
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
    def update_frame(self):
        # Lue kuva kamerasta
        ret, frame = self.cap.read()
        if ret:

            # Rajaa kuva (ROI)
            # Esimerkiksi: otetaan vain keskialue kuvaa
            height, width, _ = frame.shape
            x_start = int(width * 0.2)
            x_end = int(width * 0.8)
            y_start = int(height * 0.1)
            y_end = int(height * 0.9)
            
            # Rajattu alue kuvasta
            cropped_frame = frame[y_start:y_end, x_start:x_end]

             # Piirrä suorakulmio (x, y, leveys, korkeus) -koordinaateilla
            start_point = (100, 100)  # Suorakulmion vasen yläkulma
            end_point = (300, 300)    # Suorakulmion oikea alakulma
            color = (0, 255, 0)       # Vihreä (BGR-muodossa)
            thickness = 2             # Viivan paksuus
            
            # Piirrä suorakulmio kuvaan
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            # Muuta OpenCV BGR -muoto Qt:n RGB -muotoon
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Muuta kuva QImage-muotoon
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Aseta kuva QLabel-widgettiin
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def closeEvent(self, event):
        # Vapauta kamera suljettaessa
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.setWindowTitle("Live Camera Feed")
    window.show()
    sys.exit(app.exec_())