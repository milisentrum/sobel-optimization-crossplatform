import ctypes
import numpy as np
from PIL import Image
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import cv2
import time

class SobelApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.gray_image = None
        self.output_image = None

        # Получение абсолютного пути к DLL
        dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'sobel.dll')

        # Загрузка DLL
        try:
            self.sobel_lib = ctypes.CDLL(dll_path)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить DLL:\n{e}')
            sys.exit(1)

        # Определение типов аргументов функции apply_sobel
        self.sobel_lib.apply_sobel.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # gray_image
            ctypes.c_int,                    # width
            ctypes.c_int,                    # height
            ctypes.POINTER(ctypes.c_ubyte)   # output_image
        ]
        self.sobel_lib.apply_sobel.restype = None  # Функция возвращает void

    def init_ui(self):
        # Создание кнопок
        self.load_btn = QtWidgets.QPushButton('Загрузить изображение', self)
        self.load_folder_btn = QtWidgets.QPushButton('Загрузить папку', self)
        self.load_video_btn = QtWidgets.QPushButton('Загрузить видео', self)  # Новая кнопка для загрузки видео
        self.apply_btn = QtWidgets.QPushButton('Фильтр Собеля', self)
        self.save_btn = QtWidgets.QPushButton('Сохранить изображение', self)

        # Отключение кнопок применить и сохранить до загрузки изображения
        self.apply_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        # Создание меток для отображения изображений
        self.original_label = QtWidgets.QLabel('Исходное изображение', self)
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label = QtWidgets.QLabel('Обработанное изображение', self)
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)

        # Установка размеров меток
        self.original_label.setFixedSize(400, 400)
        self.processed_label.setFixedSize(400, 400)

        # Расположение элементов в сетке
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.load_btn, 0, 0)
        grid.addWidget(self.load_folder_btn, 0, 1)
        grid.addWidget(self.load_video_btn, 0, 2)  # Добавлена кнопка в интерфейс
        grid.addWidget(self.apply_btn, 0, 3)
        grid.addWidget(self.save_btn, 0, 4)

        grid.addWidget(self.original_label, 1, 0, 1, 2)
        grid.addWidget(self.processed_label, 1, 2, 1, 3)

        self.setLayout(grid)

        # Подключение сигналов к слотам
        self.load_btn.clicked.connect(self.load_image)
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.load_video_btn.clicked.connect(self.load_video)  # Подключение кнопки к методу
        self.apply_btn.clicked.connect(self.apply_sobel)
        self.save_btn.clicked.connect(self.save_image)

        # Настройка окна
        self.setWindowTitle('Sobel Operator Application')
        self.setGeometry(100, 100, 1000, 500)
        self.show()

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Загрузить изображение', '', 'Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*)', options=options)
        if filename:
            try:
                # Загрузка изображения с помощью Pillow
                image = Image.open(filename).convert('RGB')
                self.width, self.height = image.size
                self.original_image = image

                # Отображение изображения в QLabel
                qimage = self.pil_to_qimage(image)
                self.original_label.setPixmap(QtGui.QPixmap.fromImage(qimage).scaled(self.original_label.size(), QtCore.Qt.KeepAspectRatio))

                # Преобразование в массив NumPy и в оттенки серого
                gray = image.convert('L')  # Оттенки серого
                self.gray_image = np.array(gray, dtype=np.uint8)

                # Включение кнопки "Применить Собель"
                self.apply_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.processed_label.clear()
                self.processed_label.setText('Обработанное изображение')

                self.processing_mode = 'image'  # Установка режима обработки
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить изображение:\n{e}')

    def load_folder(self):
        options = QtWidgets.QFileDialog.Options()
        foldername = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выбрать папку', '', options=options)
        if foldername:
            # Получение списка изображений в папке
            supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = [f for f in os.listdir(foldername) if f.lower().endswith(supported_formats)]
            if not image_files:
                QtWidgets.QMessageBox.warning(self, 'Ошибка', 'В выбранной папке нет изображений.')
                return

            # Создание папки для сохранения обработанных изображений
            output_folder = os.path.join(foldername, 'sobel_output')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            progress_dialog = QtWidgets.QProgressDialog('Обработка изображений...', 'Отмена', 0, len(image_files), self)
            progress_dialog.setWindowTitle('Применение фильтра Собеля')
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

            # Обработка каждого изображения
            for i, image_file in enumerate(image_files):
                if progress_dialog.wasCanceled():
                    break
                progress_dialog.setValue(i)
                progress_dialog.setLabelText(f'Обработка {image_file}...')
                QtWidgets.QApplication.processEvents()

                image_path = os.path.join(foldername, image_file)
                try:
                    # Загрузка изображения
                    image = Image.open(image_path).convert('RGB')
                    width, height = image.size

                    # Преобразование в оттенки серого
                    gray = image.convert('L')
                    gray_image = np.array(gray, dtype=np.uint8)

                    # Подготовка буферов
                    input_buffer = gray_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

                    output_image = np.zeros_like(gray_image)
                    output_buffer = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

                    # Применение фильтра Собеля
                    self.sobel_lib.apply_sobel(input_buffer, width, height, output_buffer)

                    # Сохранение обработанного изображения
                    processed_pil = Image.fromarray(output_image)
                    processed_pil = processed_pil.convert('L')
                    output_image_path = os.path.join(output_folder, image_file)
                    processed_pil.save(output_image_path)

                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось обработать изображение {image_file}:\n{e}')

            progress_dialog.setValue(len(image_files))
            QtWidgets.QMessageBox.information(self, 'Успех', f'Обработка завершена. Обработанные изображения сохранены в {output_folder}')

    def load_video(self):
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Загрузить видео', '', 'Видео файлы (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)', options=options)
        if filename:
            try:
                # Проверка возможности открытия видео
                cap = cv2.VideoCapture(filename)
                if not cap.isOpened():
                    QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Не удалось открыть видео файл.')
                    return
                cap.release()

                self.video_path = filename

                # Отображение первого кадра видео в QLabel
                cap = cv2.VideoCapture(self.video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    qimage = self.pil_to_qimage(image)
                    self.original_label.setPixmap(QtGui.QPixmap.fromImage(qimage).scaled(self.original_label.size(), QtCore.Qt.KeepAspectRatio))

                    self.apply_btn.setEnabled(True)
                    self.save_btn.setEnabled(False)
                    self.processed_label.clear()
                    self.processed_label.setText('Обработанное видео будет сохранено после обработки')
                    self.processing_mode = 'video'  # Установка режима обработки
                else:
                    QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Не удалось прочитать первый кадр видео.')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить видео:\n{e}')

    def pil_to_qimage(self, pil_image):
        # Конвертация PIL Image в QImage
        rgb_image = pil_image.convert('RGB')
        data = rgb_image.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, rgb_image.width, rgb_image.height, QtGui.QImage.Format_RGB888)
        return qimage

    def apply_sobel(self):
        if hasattr(self, 'processing_mode'):
            if self.processing_mode == 'image':
                self.apply_sobel_to_image()
            elif self.processing_mode == 'video':
                self.apply_sobel_to_video()
            else:
                QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Неизвестный режим обработки.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Сначала загрузите изображение или видео.')

    def apply_sobel_to_image(self):
        if self.gray_image is None:
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Сначала загрузите изображение.')
            return

        try:
            # Подготовка данных для передачи в DLL
            height, width = self.gray_image.shape
            input_buffer = self.gray_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            # Создание выходного буфера
            output_image = np.zeros_like(self.gray_image)
            output_buffer = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            # Вызов функции из DLL
            self.sobel_lib.apply_sobel(input_buffer, width, height, output_buffer)

            # Сохранение результата
            self.output_image = output_image

            # Преобразование результата в QImage для отображения
            processed_pil = Image.fromarray(self.output_image)
            qimage = self.pil_to_qimage(processed_pil.convert('RGB'))
            self.processed_label.setPixmap(QtGui.QPixmap.fromImage(qimage).scaled(self.processed_label.size(), QtCore.Qt.KeepAspectRatio))

            # Включение кнопки "Сохранить"
            self.save_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось применить оператор Собеля:\n{e}')

    def apply_sobel_to_video(self):
        if not hasattr(self, 'video_path'):
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Сначала загрузите видео.')
            return

        try:
            # Открытие видео файла
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Не удалось открыть видео файл.')
                return

            # Получение параметров видео
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Выбор пути для сохранения обработанного видео
            options = QtWidgets.QFileDialog.Options()
            output_video_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить обработанное видео', '', 'Видео файлы (*.avi);;Все файлы (*)', options=options)
            if not output_video_path:
                cap.release()
                return

            # Создание VideoWriter для записи выходного видео
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), isColor=False)

            # Инициализация прогресс-бара
            progress_dialog = QtWidgets.QProgressDialog('Обработка видео...', 'Отмена', 0, frame_count, self)
            progress_dialog.setWindowTitle('Применение фильтра Собеля к видео')
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

            total_processing_time = 0  # Общее время обработки

            frame_idx = 0
            while cap.isOpened():
                if progress_dialog.wasCanceled():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                progress_dialog.setValue(frame_idx)
                progress_dialog.setLabelText(f'Обработка кадра {frame_idx+1}/{frame_count}')
                QtWidgets.QApplication.processEvents()

                # Преобразование в оттенки серого
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray_frame.shape

                # Преобразование в NumPy массив
                gray_image = np.array(gray_frame, dtype=np.uint8)

                # Подготовка буферов
                input_buffer = gray_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                output_image = np.zeros_like(gray_image)
                output_buffer = output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

                # Замер времени вызова функции из DLL
                start_time = time.perf_counter()
                self.sobel_lib.apply_sobel(input_buffer, width, height, output_buffer)
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                total_processing_time += processing_time

                # Запись обработанного кадра в выходное видео
                out.write(output_image)

                frame_idx += 1

            progress_dialog.setValue(frame_count)
            cap.release()
            out.release()

            QtWidgets.QMessageBox.information(self, 'Успех', f'Обработка завершена.\nОбработанное видео сохранено как {output_video_path}\nОбщее время обработки: {total_processing_time:.2f} секунд')
            self.save_btn.setEnabled(False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось обработать видео:\n{e}')

    def save_image(self):
        if self.output_image is None:
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Сначала примените оператор Собеля.')
            return

        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить изображение', '', 'Изображения (*.jpg *.jpeg *.png);;Все файлы (*)', options=options)
        if filename:
            try:
                # Преобразование массива NumPy в PIL Image и сохранение
                processed_pil = Image.fromarray(self.output_image)
                processed_pil = processed_pil.convert('L')  # Убедитесь, что изображение в оттенках серого
                processed_pil.save(filename)
                QtWidgets.QMessageBox.information(self, 'Успех', f'Изображение сохранено как {filename}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Не удалось сохранить изображение:\n{e}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = SobelApp()
    sys.exit(app.exec_())
