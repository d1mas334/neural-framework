-- examples/mnist_example.lua
package.path = "../?.lua;" .. package.path

local Network = require("core.network")
local Linear = require("layers.linear")
local Activation = require("layers.activation")
local CrossEntropyLoss = require("losses.cross_entropy")
local Momentum = require("optimizers.momentum")
local DataLoader = require("core.dataloader")
local Tensor = require("core.tensor")
local MNIST = require("datasets.mnist")

-- Demo settings for fast CPU runs
local DEMO_TRAIN_LIMIT = 3000
local DEMO_TEST_LIMIT = 300
local DEMO_BATCH_SIZE = 128
local DEMO_EPOCHS = 5
local RUN_FULL_TEST = false

local function progress_bar(current, total, width, loss)
    local percent = current / total
    local filled = math.floor(width * percent)
    local bar = string.rep("#", filled) .. string.rep("-", width - filled)
    local percent_str = string.format("%5.1f%%", percent * 100)
    local loss_str = loss and string.format(" - Loss: %.4f", loss) or ""
    return string.format("[%s] %s%s", bar, percent_str, loss_str)
end

print("=" .. string.rep("=", 60))
print("MNIST Example - Neural Network Training (Demo Mode)")
print("=" .. string.rep("=", 60))

print("\n[1/5] Loading MNIST dataset...")
print("  - This may take a moment...")
local mnist = MNIST.load("../data/mnist", {
    train_limit = DEMO_TRAIN_LIMIT,
    test_limit = DEMO_TEST_LIMIT,
    normalize = true
})

print("  - Raw train size: " .. #mnist.train_data)
print("  - Raw test size: " .. #mnist.test_data)

-- Trim dataset size for memory and speed
if #mnist.train_data > DEMO_TRAIN_LIMIT then
    for i = #mnist.train_data, DEMO_TRAIN_LIMIT + 1, -1 do
        mnist.train_data[i] = nil
        mnist.train_labels[i] = nil
    end
end
if #mnist.test_data > DEMO_TEST_LIMIT then
    for i = #mnist.test_data, DEMO_TEST_LIMIT + 1, -1 do
        mnist.test_data[i] = nil
        mnist.test_labels[i] = nil
    end
end
collectgarbage("collect")

print("  - Demo train size: " .. #mnist.train_data)
print("  - Demo test size: " .. #mnist.test_data)
print("  - Input dimension: " .. #mnist.train_data[1])

print("\n[2/5] Creating DataLoader...")
print("  - Batch size: " .. DEMO_BATCH_SIZE)
print("  - Shuffle: true")
local train_loader = DataLoader.new(mnist.train_data, mnist.train_labels, DEMO_BATCH_SIZE, true)
print("  - DataLoader created")

print("\n[3/5] Normalizing data...")
print("  - Skipped (dataset values are already in [0,1])")
print("  - Data normalization skipped")

print("\n[4/5] Creating neural network...")
print("  Architecture:")
print("    - Input: 784")
print("    - Hidden: 64 (ReLU)")
print("    - Output: 10 (Softmax)")
local params = 784 * 64 + 64 + 64 * 10 + 10
print("  - Trainable params: " .. params)

local network = Network.new({
    Linear.new(784, 64),
    Activation.new("relu"),
    Linear.new(64, 10),
    Activation.new("softmax")
})
print("  - Network created")

print("\n[5/5] Setting up training...")
local loss_fn = CrossEntropyLoss.new()
local optimizer = Momentum.new(0.01, 0.9)
print("  - Loss: Cross Entropy")
print("  - Optimizer: Momentum SGD")
print("  - Learning rate: 0.01")
print("  - Momentum: 0.9")

print("\n" .. string.rep("=", 60))
print("STARTING TRAINING")
print(string.rep("=", 60))
print("\nTraining network for " .. DEMO_EPOCHS .. " epochs...\n")

local start_time = os.clock()
local total_batches = math.ceil(#mnist.train_data / DEMO_BATCH_SIZE)

for epoch = 1, DEMO_EPOCHS do
    local total_loss = 0
    local batch_count = 0
    local epoch_start = os.clock()

    train_loader:reset()
    io.write(string.format("Epoch %d/%d\n", epoch, DEMO_EPOCHS))

    for batch in train_loader:iter() do
        batch_count = batch_count + 1

        local predictions = network:forward(batch.input)
        local loss = loss_fn:forward(predictions, batch.target)
        total_loss = total_loss + loss

        local grad = loss_fn:backward()
        network:backward(grad)

        network:update_params(optimizer)
        network:zero_grad()

        if batch_count % 5 == 0 or batch_count == total_batches then
            local bar = progress_bar(batch_count, total_batches, 40, loss)
            io.write("\r  " .. bar)
            io.flush()
        end
    end

    local avg_loss = total_loss / batch_count
    local epoch_time = os.clock() - epoch_start
    print(string.format("\r  - Epoch %d complete | Avg Loss: %.4f | Time: %.2fs", epoch, avg_loss, epoch_time))
    print()
end

local end_time = os.clock()
print("\n" .. string.rep("=", 60))
print("TRAINING COMPLETED")
print(string.rep("=", 60))
print(string.format("Total training time: %.2f sec (%.2f min)", end_time - start_time, (end_time - start_time) / 60))

print("\n" .. string.rep("=", 60))
print("TESTING")
print(string.rep("=", 60))

local function predict_class(sample)
    local t = Tensor.new()
    t.data = {sample}
    t.shape = {1, 784}
    local prediction = network:predict(t)

    local max_val = -math.huge
    local max_idx = 1
    for j, val in ipairs(prediction.data[1]) do
        if val > max_val then
            max_val = val
            max_idx = j
        end
    end
    return max_idx - 1, max_val
end

print("\nFirst sample:")
local p0, conf0 = predict_class(mnist.test_data[1])
print(string.format("  - Predicted: %d (conf %.4f)", p0, conf0))
print(string.format("  - Actual: %d", mnist.test_labels[1]))

print("\nBatch testing (first 100 samples):")
local correct = 0
local total = math.min(100, #mnist.test_data)
for i = 1, total do
    local predicted = predict_class(mnist.test_data[i])
    local actual = mnist.test_labels[i]
    if predicted == actual then
        correct = correct + 1
    end
    print(string.format("  Sample %2d: predicted=%d actual=%d", i, predicted, actual))
end
print(string.format("Accuracy on first %d samples: %.1f%% (%d/%d)", total, (correct / total) * 100, correct, total))

if RUN_FULL_TEST then
    print("\nFull testing on all demo test samples...")
    local full_correct = 0
    local full_total = #mnist.test_data
    for i = 1, full_total do
        local predicted = predict_class(mnist.test_data[i])
        if predicted == mnist.test_labels[i] then
            full_correct = full_correct + 1
        end
    end
    print(string.format("Full-test accuracy: %.2f%% (%d/%d)", (full_correct / full_total) * 100, full_correct, full_total))
else
    print("\nFull testing skipped in demo mode.")
end

print("\n" .. string.rep("=", 60))
print("EXAMPLE COMPLETED")
print(string.rep("=", 60))
