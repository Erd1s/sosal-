import os
import time
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from nets import DGNLNet
from misc import check_mkdir

# Конфигурация CPU
device = torch.device("cpu")
torch.manual_seed(2019)

# Параметры модели
config = {
    'ckpt_path': './ckpt',  # Путь к весам модели
    'exp_name': 'DGNLNet',  # Название эксперимента
    'snapshot': '40000',  # Имя файла с весами
    'input_dir': r'C:\Users\erd1s\Downloads\DGNL-Net-main\imagesname',  # Путь к тестовым изображениям
    'output_dir': './results',  # Папка для сохранения результатов
    'img_extensions': ('.png', '.jpg', '.jpeg')  # Поддерживаемые форматы изображений
}

# Предобработка изображений
transform = transforms.Compose([
    transforms.Resize([1024, 512]),  # Размер, ожидаемый моделью
    transforms.ToTensor()
])


def load_model():
    """Загрузка предобученной модели"""
    model = DGNLNet().to(device)
    model_path = os.path.join(config['ckpt_path'], config['exp_name'], f"{config['snapshot']}.pth")

    print(f"Загрузка модели из {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    return model


def process_image(model, img_path, output_path):
    """Обработка одного изображения"""
    try:
        # Загрузка и преобразование изображения
        img = Image.open(img_path).convert('RGB')
        original_size = img.size

        # Подготовка входных данных
        img_tensor = Variable(transform(img).unsqueeze(0)).to(device)

        # Обработка моделью
        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor)
            process_time = time.time() - start_time

        # Преобразование результата
        result = transforms.Resize(original_size)(transforms.ToPILImage()(output.data.squeeze(0).cpu()))
        result.save(output_path)

        return process_time

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return 0


def main():
    # Проверка и создание директорий
    check_mkdir(config['output_dir'])
    output_folder = os.path.join(config['output_dir'], f"{config['exp_name']}_results")
    check_mkdir(output_folder)

    # Загрузка модели
    model = load_model()

    # Получение списка изображений
    img_list = [
        f for f in os.listdir(config['input_dir'])
        if f.lower().endswith(config['img_extensions'])
    ]

    print(f"Найдено {len(img_list)} изображений для обработки")

    # Обработка изображений
    total_time = 0
    for idx, img_name in enumerate(img_list, 1):
        img_path = os.path.join(config['input_dir'], img_name)
        output_path = os.path.join(output_folder, img_name)

        process_time = process_image(model, img_path, output_path)
        total_time += process_time

        print(f"Обработано: {idx}/{len(img_list)} | "
              f"Файл: {img_name} | "
              f"Время: {process_time:.2f} сек | "
              f"Среднее время: {total_time / idx:.2f} сек")

    print(f"\nОбработка завершена! Результаты сохранены в {output_folder}")
    print(f"Общее время: {total_time:.2f} сек | "
          f"Среднее время на изображение: {total_time / len(img_list):.2f} сек")


if __name__ == '__main__':
    main()