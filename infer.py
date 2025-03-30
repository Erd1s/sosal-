import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from nets import basic, depth_predciton, basic_NL, DGNLNet
from config import test_raincityscapes_path
from misc import check_mkdir

device = torch.device("cpu")  # Используем только процессор (CPU)

ckpt_path = './ckpt'  # Путь к сохраненным моделям
exp_name = 'DGNLNet'  # Название эксперимента
args = {
    'snapshot': '40000',  # Имя сохраненной модели
    'depth_snapshot': ''  # Имя модели для глубины (если есть)
}

# Преобразования для входных изображений
transform = transforms.Compose([
    transforms.Resize([512, 1024]),  # Изменение размера
    transforms.ToTensor()])  # Преобразование в тензор

# Путь к тестовым изображениям
root = r'C:\Users\erd1s\Downloads\DGNL-Net-main\imagesname'

to_pil = transforms.ToPILImage()  # Для обратного преобразования в изображение


def main():
    # Загружаем модель на CPU
    net = DGNLNet().to(device)

    if len(args['snapshot']) > 0:
        print('Загружаем сохраненную модель "%s" для тестирования' % args['snapshot'])
        net.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                map_location=device  # Загружаем веса на CPU
            )
        )

    net.eval()  # Переключаем модель в режим оценки
    avg_time = 0  # Для подсчета среднего времени обработки

    with torch.no_grad():  # Отключаем вычисление градиентов
        img_list = [img_name for img_name in os.listdir(root)]  # Список изображений

        for idx, img_name in enumerate(img_list):
            # Создаем директорию для сохранения результатов
            check_mkdir(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['snapshot'])))
            if len(args['depth_snapshot']) > 0:
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['depth_snapshot'])))

            # Открываем и преобразуем изображение
            img = Image.open(os.path.join(root, img_name)).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).to(device)

            start_time = time.time()  # Замер времени начала обработки

            res = net(img_var)  # Обрабатываем изображение моделью

            avg_time = avg_time + time.time() - start_time  # Обновляем среднее время

            print('Обработка: %d / %d, среднее время: %.5f' % (idx + 1, len(img_list), avg_time / (idx + 1)))

            # Преобразуем результат обратно в изображение
            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))

            # Сохраняем результат
            result.save(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
                    exp_name, args['snapshot']), img_name))


if __name__ == '__main__':
    main()