import cv2
import os
import time

# Укажите путь к видеофайлу
video_path = '5.mkv'

# Укажите путь к каталогу, где будут сохранены изображения
output_directory = './sm_dataset'

# Установите интервал между извлекаемыми кадрами (в секундах)
frame_interval = 0.1

def extract_frames(video_path, output_directory, interval):
    # Проверяем, существует ли указанный каталог, и создаем его, если нет
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    # Переменная для отслеживания текущего времени
    current_time = time.time()

    # Переменная для отслеживания номера кадра
    frame_number = 2241


    while True:
        # Читаем следующий кадр из видео
        ret, frame = cap.read()

        # Проверяем, удалось ли прочитать кадр
        if not ret:
            break

        # Проверяем, прошло ли достаточно времени с предыдущего сохранения кадра
        if time.time() - current_time >= interval:
            # Изменяем размер изображения на 256x256
            resized_frame = cv2.resize(frame, (256, 256))

            # Сохраняем изображение в указанный каталог
            output_path = os.path.join(output_directory, f'frame_{frame_number}.jpg')
            cv2.imwrite(output_path, resized_frame)

            # Обновляем текущее время и увеличиваем номер кадра
            current_time = time.time()
            frame_number += 1

    # Закрываем видеофайл
    cap.release()

if __name__ == '__main__':
    extract_frames(video_path, output_directory, frame_interval)
