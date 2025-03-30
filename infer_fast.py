import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from nets import DGNLNet_fast
from config import test_raincityscapes_path
from misc import check_mkdir

# Устанавливаем использование CPU
device = torch.device("cpu")
torch.manual_seed(2019)

ckpt_path = './ckpt'
exp_name = 'DGNLNet_fast'
args = {
    'snapshot': '60000',
    'depth_snapshot': ''
}

transform = transforms.Compose([
    transforms.Resize([512, 1024]),
    transforms.ToTensor()
])

root = r'C:\Users\erd1s\Downloads\DGNL-Net-main\imagesname'
to_pil = transforms.ToPILImage()


def main():
    # Загружаем модель на CPU
    net = DGNLNet_fast().to(device)

    if len(args['snapshot']) > 0:
        print(f'Загружаем модель "{args["snapshot"]}" для тестирования')
        # Загружаем веса с указанием устройства - CPU
        net.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                map_location=device
            )
        )

    net.eval()
    avg_time = 0

    with torch.no_grad():
        # Получаем список изображений (только с поддерживаемыми расширениями)
        img_list = [
            img_name for img_name in os.listdir(root)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        for idx, img_name in enumerate(img_list):
            # Создаем директорию для результатов
            output_dir = os.path.join(
                ckpt_path, exp_name,
                f'({exp_name}) prediction_{args["snapshot"]}'
            )
            check_mkdir(output_dir)

            # Загружаем и преобразуем изображение
            img_path = os.path.join(root, img_name)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            # Переносим тензор на CPU
            img_var = Variable(transform(img).unsqueeze(0)).to(device)

            start_time = time.time()

            # Обработка изображения
            res = net(img_var)

            # Рассчитываем время обработки
            process_time = time.time() - start_time
            avg_time += process_time

            print(f'Обработка: {idx + 1}/{len(img_list)}, '
                  f'текущее время: {process_time:.5f} сек, '
                  f'среднее время: {avg_time / (idx + 1):.5f} сек')

            # Преобразуем результат и сохраняем
            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
            result.save(os.path.join(output_dir, img_name))


if __name__ == '__main__':
    main()