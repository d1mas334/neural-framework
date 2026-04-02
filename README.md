# Neural Framework (Lua)

Лабораторная работа по курсу **"Искусственный интеллект"**  
Тема: **Создание своего нейросетевого фреймворка**

## Команда

| ФИО | Роль в проекте | Оценка |
|---|---|---|
| Иванопуло | Руководил |  |
| Петропуло | Программировал |  |
| Гин Ку Син | Программировал оптимизаторы |  |
| Галкина | Писала отчёт |  |

## Что реализовано

- Создание многослойной нейросети перечислением слоев (`Network.new({ ... })`).
- Базовые абстракции фреймворка: `Tensor`, `Layer`, `Network`.
- Работа с данными: `DataLoader` (`minibatching`, `shuffle`, `map`).
- Оптимизаторы:
  - `SGD`
  - `Momentum SGD`
  - `Adam`
- Передаточные функции:
  - `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`
- Функции потерь:
  - `CrossEntropy` (multi-class классификация)
  - `BinaryCrossEntropy` (binary классификация)
  - `MSE` (регрессия)
- Обучение "в несколько строк" через `network:train(...)` или явный train loop в примерах.

## Структура проекта

```text
core/         # tensor, network, layer, dataloader
layers/       # linear, activation, dropout
losses/       # cross_entropy, binary_cross_entropy, mse
optimizers/   # sgd, momentum, adam
datasets/     # mnist, iris, generated_vs_real
examples/     # mnist_example, iris_example, generated_vs_real_example
tools/        # утилиты подготовки датасетов
```

## Требования

- Lua 5.1+ (проверено на Lua 5.1)
- Для подготовки image-датасетов:
  - Python 3.9+
  - пакеты: `pillow`, `datasets`

## Быстрый старт

### 1) MNIST (реальные IDX-файлы)

Ожидаемые файлы в `data/mnist`:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

Запуск:

```powershell
cd D:\GitHub\neural-framework\examples
lua mnist_example.lua
```

### 2) Iris (demo example)

```powershell
cd D:\GitHub\neural-framework\examples
lua iris_example.lua 40 16 0.01
```

Аргументы: `epochs batch_size learning_rate`.

### 3) Generated vs Real (binary classification)

#### Подготовка CSV из папок с картинками

```powershell
cd D:\GitHub\neural-framework
python -m pip install pillow datasets

python tools\build_generated_vs_real_csv.py `
  --real-dir D:\GitHub\neural-framework\data\binary_images\real `
  --fake-dir D:\GitHub\neural-framework\data\binary_images\fake `
  --out D:\GitHub\neural-framework\data\generated_vs_real.csv `
  --size 28 `
  --max-per-class 2000
```

Запуск обучения:

```powershell
cd D:\GitHub\neural-framework\examples
lua generated_vs_real_example.lua ../data/generated_vs_real.csv 12 128 adam
```

Аргументы: `csv_path epochs batch_size optimizer_name`.

## Формат данных для binary classification

`generated_vs_real_example.lua` ожидает CSV:

```csv
label,p1,p2,...,pN
0,0.12,0.34,...
1,0.91,0.07,...
```

- `label`: `0` (real), `1` (generated)
- признаки `p1..pN`: числовые (в проекте это векторизованные пиксели)

## Важные замечания

- Для `CrossEntropy` используются **индексы классов** (`0..C-1`), не one-hot.
- Примеры в `examples/` предполагают запуск из директории `examples` (из-за `package.path = "../?.lua;" .. package.path`).
- Большие датасеты в `data/` в репозитории лучше не коммитить в публичный remote без необходимости.

## Соответствие заданию (кратко)

- Многослойная сеть перечислением слоев: **да**
- Data utils (`map`, minibatch, shuffle): **да**
- Не менее 3 оптимизаторов: **да** (`SGD`, `Momentum`, `Adam`)
- Несколько функций активации и потерь для классификации/регрессии: **да**
- Обучение в несколько строк: **да**
- 2-3 примера: **да** (`MNIST`, `Iris`, `Generated vs Real`)
- README: **да**

## Видео (5 минут)

В видео рекомендуется показать:

1. Архитектуру фреймворка (`core`, `layers`, `losses`, `optimizers`).
2. Короткий запуск `mnist_example.lua` и `iris_example.lua`.
3. Нестандартный кейс `generated_vs_real_example.lua` (детекция AI-generated изображений).
