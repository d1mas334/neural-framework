# Neural Framework (Lua)

Лабораторная работа по курсу "Искусственный интеллект"  
Тема: **Создание своего нейросетевого фреймворка**

## Состав команды

| ФИО | Роль в проекте | Оценка |
|---|---|---|
| Иванопуло | Руководил |  |
| Петропуло | Программировал |  |
| Гин Ку Син | Программировал оптимизаторы |  |
| Галкина | Писала отчёт |  |

## Цель работы

Реализация собственного фреймворка для обучения полносвязных нейросетей с поддержкой:

- построения многослойной сети;
- загрузки и обработки данных;
- нескольких оптимизаторов;
- функций активации и функций потерь;
- примеров обучения на классических задачах.

## Реализованные компоненты

### 1. Ядро фреймворка (`core`)

- `tensor.lua` — базовые операции над тензорами;
- `layer.lua` — базовый интерфейс слоя;
- `network.lua` — последовательная многослойная сеть, `forward/backward`, обновление параметров;
- `dataloader.lua` — `minibatching`, `shuffle`, `map`, итератор батчей.

### 2. Слои (`layers`)

- `linear.lua` — полносвязный слой;
- `activation.lua` — `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`;
- `dropout.lua` — dropout-слой.

### 3. Функции потерь (`losses`)

- `cross_entropy.lua` — многоклассовая классификация;
- `binary_cross_entropy.lua` — бинарная классификация;
- `mse.lua` — среднеквадратичная ошибка (регрессия).

### 4. Оптимизаторы (`optimizers`)

- `sgd.lua` — SGD;
- `momentum.lua` — Momentum SGD;
- `adam.lua` — Adam.

### 5. Датасеты (`datasets`)

- `mnist.lua` — загрузка реального MNIST (IDX-файлы);
- `iris.lua` — iris-like датасет для классификации на 3 класса;
- `generated_vs_real.lua` — загрузка бинарного CSV (`label + признаки`).

### 6. Примеры (`examples`)

- `mnist_example.lua` — обучение MLP на MNIST;
- `iris_example.lua` — обучение сети на Iris;
- `generated_vs_real_example.lua` — бинарная классификация real vs generated.

### 7. Утилиты подготовки данных (`tools`)

- `build_generated_vs_real_csv.py` — сборка CSV из папок `real/fake`;
- `download_defactify_subset.py` — загрузка subset датасета Defactify;
- `download_cocoai_subset.py` — загрузка subset COCO_AI;
- `prepare_binary_image_subset.py` — подготовка бинарной выборки из сырой структуры.

## Соответствие пунктам задания

- Создание многослойной нейросети перечислением слоёв — **выполнено**.
- Набор функций для работы с данными (`map`, minibatching, shuffle) — **выполнено**.
- Не менее 3 оптимизаторов — **выполнено** (`SGD`, `Momentum`, `Adam`).
- Передаточные функции и функции потерь для классификации и регрессии — **выполнено**.
- Обучение нейросети в несколько строк с конфигурированием — **выполнено**.
- 2-3 примера использования на классических задачах — **выполнено** (`MNIST`, `Iris`, `Generated vs Real`).
- Документация в `README.md` — **выполнено**.

## Особенности проекта

- Проект реализован на языке **Lua** (нестандартный выбор для задач нейросетей).
- Фреймворк написан с нуля: собственные `Tensor`, `Network`, `DataLoader`, слои, лоссы и оптимизаторы.
- Помимо классических задач (`MNIST`, `Iris`) добавлен отдельный практический кейс:
  **бинарная классификация изображений `AI / not AI`** (`generated vs real`).

## Структура проекта

```text
core/
layers/
losses/
optimizers/
datasets/
examples/
tools/
data/
README.md
```

## Пошаговый гайд

### Шаг 1. Установить инструменты

Открыть **PowerShell** и выполнить:

```powershell
winget install --id Git.Git -e
winget install --id Lua.Lua -e
```

Закрыть и заново открыть PowerShell.

Проверить, что всё установилось:

```powershell
git --version
lua -v
```

### Шаг 2. Скачать репозиторий с GitHub

```powershell
cd D:\
mkdir GitHub -ErrorAction SilentlyContinue
cd D:\GitHub
git clone <URL_ВАШЕГО_REPO_ИЗ_GITHUB_CLASSROOM_ИЛИ_PUBLIC_REPO>
cd neural-framework
```

### Шаг 3. Запустить примеры работоспособности

```powershell
cd D:\GitHub\neural-framework\examples
lua iris_example.lua 30 16 0.01
lua mnist_example.lua
lua generated_vs_real_example.lua ../data/generated_vs_real.csv 12 128 adam
```

## Запуск примеров

### MNIST

```powershell
cd D:\GitHub\neural-framework\examples
lua mnist_example.lua
```

### Iris

```powershell
cd D:\GitHub\neural-framework\examples
lua iris_example.lua 30 16 0.01
```

Параметры: `epochs batch_size learning_rate`.

### Generated vs Real

```powershell
cd D:\GitHub\neural-framework\examples
lua generated_vs_real_example.lua ../data/generated_vs_real.csv 12 128 adam
```

Параметры: `csv_path epochs batch_size optimizer`.

## Формат входного CSV для binary classification

```csv
label,p1,p2,...,pN
0,0.12,0.34,...
1,0.91,0.07,...
```

- `label`: `0` — real, `1` — generated.
- `p1..pN`: числовые признаки.

## Примечание по меткам

Для `CrossEntropy` используются индексы классов (`0..C-1`), без one-hot кодирования.

## Кратко по датасетам

### MNIST

- Используется реальный набор рукописных цифр (IDX-файлы в `data/mnist`).
- В примере применяется MLP-сеть и `CrossEntropy` для многоклассовой классификации.
- Скрипт: `examples/mnist_example.lua`.

### Iris

- Используется iris-like датасет для задачи классификации на 3 класса.
- В примере используется MLP + `CrossEntropy` + `Adam`.
- Скрипт: `examples/iris_example.lua`.

### Generated vs Real (AI / not AI)

- Используется бинарный датасет изображений (`real` и `generated`), собранный в CSV-признаки.
- В примере используется MLP + `BinaryCrossEntropy` + `Adam`.
- Добавлен практический кейс определения, сгенерировано изображение ИИ или нет.
- Скрипт: `examples/generated_vs_real_example.lua`.

## Материалы для защиты

- Репозиторий фреймворка с исходным кодом;
- Примеры запуска (`MNIST`, `Iris`, `Generated vs Real`);
- Метрики качества из выводов скриптов;
- Видео-презентация (5 минут).
