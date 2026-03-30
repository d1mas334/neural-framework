-- examples/mnist_example.lua
package.path = "../?.lua;" .. package.path

local Network = require("core.network")
local Linear = require("layers.linear")
local Activation = require("layers.activation")
local CrossEntropyLoss = require("losses.cross_entropy")
local SGD = require("optimizers.sgd")
local Momentum = require("optimizers.momentum")
local DataLoader = require("core.dataloader")
local MNIST = require("datasets.mnist")

-- Функция для создания прогресс-бара
local function progress_bar(current, total, width, loss)
    local percent = current / total
    local filled = math.floor(width * percent)
    local bar = string.rep("█", filled) .. string.rep("░", width - filled)
    local percent_str = string.format("%5.1f%%", percent * 100)
    local loss_str = loss and string.format(" - Loss: %.4f", loss) or ""
    return string.format("[%s] %s%s", bar, percent_str, loss_str)
end

print("=" .. string.rep("=", 60))
print("MNIST Example - Neural Network Training")
print("=" .. string.rep("=", 60))

-- Загрузка данных
print("\n[1/5] Loading MNIST dataset...")
print("  - This may take a moment...")
local mnist = MNIST.load("data/mnist")

print("  ✓ Train data size: " .. #mnist.train_data)
print("  ✓ Train labels size: " .. #mnist.train_labels)
print("  ✓ Test data size: " .. #mnist.test_data)
print("  ✓ Test labels size: " .. #mnist.test_labels)
print("  ✓ Input dimension: " .. #mnist.train_data[1])

-- Создание DataLoader
print("\n[2/5] Creating DataLoader...")
print("  - Batch size: 64")
print("  - Shuffle: true")
local train_loader = DataLoader.new(
    mnist.train_data, 
    mnist.train_labels, 
    64,  -- batch_size
    true -- shuffle
)
print("  ✓ DataLoader created")

-- Нормализация данных
print("\n[3/5] Normalizing data...")
print("  - Scaling pixel values from [0,255] to [0,1]")
train_loader:map(function(x)
    local normalized = {}
    for i, val in ipairs(x) do
        normalized[i] = val / 255.0
    end
    return normalized
end)
print("  ✓ Data normalized")

-- Создание сети
print("\n[4/5] Creating neural network...")
print("  Architecture:")
print("    - Input layer: 784 neurons")
print("    - Hidden layer 1: 128 neurons (ReLU)")
print("    - Hidden layer 2: 64 neurons (ReLU)")
print("    - Output layer: 10 neurons (Softmax)")
print("  Total parameters: 784*128 + 128 + 128*64 + 64 + 64*10 + 10 =")
local params = 784*128 + 128 + 128*64 + 64 + 64*10 + 10
print("    " .. params .. " trainable parameters")

local network = Network.new({
    Linear.new(784, 128),      -- Входной слой (28x28=784) -> 128 нейронов
    Activation.new("relu"),     -- ReLU активация
    Linear.new(128, 64),        -- Скрытый слой 128 -> 64
    Activation.new("relu"),     -- ReLU активация
    Linear.new(64, 10),         -- Выходной слой 64 -> 10 (классы)
    Activation.new("softmax")   -- Softmax для вероятностей
})
print("  ✓ Network created")

-- Функция потерь
print("\n[5/5] Setting up training...")
local loss_fn = CrossEntropyLoss.new()
print("  - Loss function: Cross Entropy Loss")

-- Оптимизатор
local optimizer = Momentum.new(0.01, 0.9)
print("  - Optimizer: Momentum SGD")
print("    * Learning rate: 0.01")
print("    * Momentum: 0.9")

print("\n" .. string.rep("=", 60))
print("STARTING TRAINING")
print(string.rep("=", 60))

-- Обучение с прогресс-баром
print("\nTraining network for 10 epochs...\n")
local start_time = os.clock()

-- Подсчет общего количества батчей
local total_batches = math.ceil(#mnist.train_data / 64)

for epoch = 1, 10 do
    local total_loss = 0
    local batch_count = 0
    local epoch_start = os.clock()
    
    train_loader:reset()
    
    -- Прогресс-бар для эпохи
    io.write(string.format("Epoch %d/10\n", epoch))
    
    for batch in train_loader:iter() do
        batch_count = batch_count + 1
        
        -- Forward pass
        local predictions = network:forward(batch.input)
        local loss = loss_fn:forward(predictions, batch.target)
        total_loss = total_loss + loss
        
        -- Backward pass
        local grad = loss_fn:backward()
        network:backward(grad)
        
        -- Update parameters
        network:update_params(optimizer)
        
        -- Zero gradients
        network:zero_grad()
        
        -- Обновление прогресс-бара каждые 10 батчей
        if batch_count % 10 == 0 or batch_count == total_batches then
            local bar = progress_bar(batch_count, total_batches, 50, loss)
            io.write("\r  " .. bar)
            io.flush()
        end
    end
    
    local avg_loss = total_loss / batch_count
    local epoch_time = os.clock() - epoch_start
    
    -- Вывод информации об эпохе
    print(string.format("\r  ✓ Epoch %d completed - Avg Loss: %.4f - Time: %.2fs", 
          epoch, avg_loss, epoch_time))
    print()
end

local end_time = os.clock()

print("\n" .. string.rep("=", 60))
print("TRAINING COMPLETED")
print(string.rep("=", 60))
print(string.format("Total training time: %.2f seconds (%.2f minutes)", 
      end_time - start_time, (end_time - start_time) / 60))

-- Тестирование на первом примере
print("\n" .. string.rep("=", 60))
print("TESTING")
print(string.rep("=", 60))

print("\nTesting on first test sample...")
local test_input = require("core.tensor").new()
test_input.data = {mnist.test_data[1]}
test_input.shape = {1, 784}

print("  - Forward pass...")
local prediction = network:predict(test_input)

print("\n  - Prediction probabilities:")
local max_val = -math.huge
local max_idx = 1
for i, val in ipairs(prediction.data[1]) do
    local bar = string.rep("█", math.floor(val * 50))
    print(string.format("    Class %d: %.4f %s", i-1, val, bar))
    if val > max_val then
        max_val = val
        max_idx = i
    end
end

print("\n  - Result:")
print(string.format("    Predicted class: %d", max_idx - 1))
print(string.format("    True class: %d", mnist.test_labels[1]))

if max_idx - 1 == mnist.test_labels[1] then
    print("    ✓ Correct prediction!")
else
    print("    ✗ Wrong prediction")
end

-- Дополнительное тестирование на 10 примерах
print("\n" .. string.rep("=", 60))
print("BATCH TESTING (first 10 test samples)")
print(string.rep("=", 60))

local correct = 0
local total = 10

for i = 1, total do
    local test_input = require("core.tensor").new()
    test_input.data = {mnist.test_data[i]}
    test_input.shape = {1, 784}
    
    local prediction = network:predict(test_input)
    
    local max_val = -math.huge
    local max_idx = 1
    for j, val in ipairs(prediction.data[1]) do
        if val > max_val then
            max_val = val
            max_idx = j
        end
    end
    
    local predicted = max_idx - 1
    local actual = mnist.test_labels[i]
    
    if predicted == actual then
        correct = correct + 1
        print(string.format("  Sample %2d: Predicted=%d, Actual=%d ✓", i, predicted, actual))
    else
        print(string.format("  Sample %2d: Predicted=%d, Actual=%d ✗", i, predicted, actual))
    end
end

print(string.format("\nAccuracy on first %d test samples: %.1f%% (%d/%d)", 
      total, (correct/total)*100, correct, total))

-- Тестирование на всех тестовых данных (опционально)
print("\n" .. string.rep("=", 60))
print("FULL TESTING (all test samples)")
print(string.rep("=", 60))
print("  - This may take a moment...")

local full_correct = 0
local full_total = #mnist.test_data
local test_progress_width = 40

io.write("  Testing: ")
io.flush()

for i = 1, full_total do
    local test_input = require("core.tensor").new()
    test_input.data = {mnist.test_data[i]}
    test_input.shape = {1, 784}
    
    local prediction = network:predict(test_input)
    
    local max_val = -math.huge
    local max_idx = 1
    for j, val in ipairs(prediction.data[1]) do
        if val > max_val then
            max_val = val
            max_idx = j
        end
    end
    
    local predicted = max_idx - 1
    local actual = mnist.test_labels[i]
    
    if predicted == actual then
        full_correct = full_correct + 1
    end
    
    -- Показываем прогресс каждые 100 примеров
    if i % 100 == 0 or i == full_total then
        local percent = i / full_total
        local filled = math.floor(test_progress_width * percent)
        local bar = string.rep("█", filled) .. string.rep("░", test_progress_width - filled)
        io.write("\r  Testing: [" .. bar .. "] " .. string.format("%5.1f%%", percent * 100))
        io.flush()
    end
end

print()
local full_accuracy = (full_correct / full_total) * 100
print(string.format("\n✓ Final accuracy on all %d test samples: %.2f%% (%d/%d)", 
      full_total, full_accuracy, full_correct, full_total))

print("\n" .. string.rep("=", 60))
print("EXAMPLE COMPLETED")
print(string.rep("=", 60))