# Проект: Нейросети для перевода изображений в стиль мультфильма о Человеке-пауке

## Введение
Работа представляет собой финальный проект первого семестра Deep Learning School (осень 2023). Проект представляет собой реализацию CycleGAN сети глубокого обучения для решения image2image задачи. Работа сосредоточена на стилизации изображений, чтобы они соответствовали визуальному стилю, заимствованному из конкретных видеороликов, таких как мультфильмы, фильмы или стримы видеоигр.

Была разработана модель CycleGAN, которая позволяет нам переносить стили из видео в изображения. В проекте представлена архитектура, позволяющая выбирать тип генератора (ResNet и UNet), а также алгоритм формирования датасета, использующий извлечение кадров из видеороликов. Наша модель решает задачу стилизации исходного изображения в целевой стиль.

## Архитектура модели
### Генераторы:
- Стандартный генератор (ResNet-based): Основан на архитектуре ResNet.
- Дополнительный генератор (UNet-based): Использует архитектуру UNet.
#### ResNet Генератор:
- Пользователь может выбрать тип Residual блока при создании:
- InstanceNorm 2D (по умолчанию).
- BatchNorm2D.
### Дискриминатор:
- Реализован в соответствии с оригинальной статьей, используя архитектуру PatchGAN.
### Создание датасетов
- Для формирования необходимых наборов изображений из видеофайлов разработан скрипт, который позволяет создавать датасеты, необходимые для обучения сети.

## Использованные данные
- Серии мультфильма о Человеке-пауке (1994 год):
    Изображения из этих серий использовались для обучения сети и являлись целевым стилем.
- Monet2Image датасет:
    Для разнообразия использовались нестилизованные изображения из датасета оригинальной статьи.
## Задача
Наша нейросеть решает задачу перевода реальных изображений в стиль мультфильма о Человеке-пауке.

## Особенности работы
Метод создания датасетов из видео является важным дополнением к работе.
Для проверки результатов работы создан Telegram бот. Инструкции по запуску доступны в папке "tg_bot".
Для проверки бота необходим BOT_TOKEN, который можно получить в Telegram у бота BotFather.
Обязательным условием является наличие на компьютере Docker, так как для простоты развертывания бот разворачивается и работает в контейнере.

## Выводы по работе