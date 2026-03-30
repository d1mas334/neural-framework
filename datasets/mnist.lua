-- datasets/mnist.lua
local MNIST = {}

function MNIST.load(data_path)
    -- Упрощенная загрузка MNIST
    -- В реальном проекте нужно загрузить настоящие данные
    print("Loading MNIST dataset from " .. (data_path or "default path"))
    
    local train_data = {}
    local train_labels = {}
    local test_data = {}
    local test_labels = {}
    
    -- Здесь должна быть реальная загрузка данных
    -- Для примера создаем случайные данные
    
    math.randomseed(42)
    
    -- Создаем 60000 тренировочных примеров
    for i = 1, 60000 do
        local sample = {}
        for j = 1, 784 do -- 28x28
            table.insert(sample, math.random() * 255 / 255)
        end
        table.insert(train_data, sample)
        table.insert(train_labels, math.random(0, 9))
    end
    
    -- Создаем 10000 тестовых примеров
    for i = 1, 10000 do
        local sample = {}
        for j = 1, 784 do
            table.insert(sample, math.random() * 255 / 255)
        end
        table.insert(test_data, sample)
        table.insert(test_labels, math.random(0, 9))
    end
    
    return {
        train_data = train_data,
        train_labels = train_labels,
        test_data = test_data,
        test_labels = test_labels
    }
end

return MNIST