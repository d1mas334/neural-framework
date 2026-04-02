package.path = "../?.lua;" .. package.path

local Network = require("core.network")
local Linear = require("layers.linear")
local Activation = require("layers.activation")
local CrossEntropyLoss = require("losses.cross_entropy")
local Adam = require("optimizers.adam")
local DataLoader = require("core.dataloader")
local Tensor = require("core.tensor")
local Iris = require("datasets.iris")

local EPOCHS = tonumber(arg[1]) or 40
local BATCH_SIZE = tonumber(arg[2]) or 16
local LR = tonumber(arg[3]) or 0.01

print("=" .. string.rep("=", 60))
print("Iris Example - Multi-class Classification")
print("=" .. string.rep("=", 60))

local ds = Iris.load({
    test_ratio = 0.2,
    shuffle = true,
    seed = 42
})

print(string.format("Dataset: train=%d test=%d input_dim=%d classes=%d",
    #ds.train_data, #ds.test_data, ds.input_dim, ds.classes))

local train_loader = DataLoader.new(ds.train_data, ds.train_labels, BATCH_SIZE, true)

local network = Network.new({
    Linear.new(4, 16),
    Activation.new("relu"),
    Linear.new(16, 3),
    Activation.new("softmax")
})

local loss_fn = CrossEntropyLoss.new()
local optimizer = Adam.new(LR, 0.9, 0.999, 1e-8)

print(string.format("Training: epochs=%d batch=%d lr=%.4f", EPOCHS, BATCH_SIZE, LR))

for epoch = 1, EPOCHS do
    local total_loss = 0
    local batches = 0
    train_loader:reset()

    for batch in train_loader:iter() do
        local pred = network:forward(batch.input)
        local loss = loss_fn:forward(pred, batch.target)
        total_loss = total_loss + loss
        batches = batches + 1

        local grad = loss_fn:backward()
        network:backward(grad)
        network:update_params(optimizer)
        network:zero_grad()
    end

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS then
        print(string.format("Epoch %d/%d | avg loss: %.6f", epoch, EPOCHS, total_loss / batches))
    end
end

local function predict_class(sample)
    local t = Tensor.new()
    t.data = {sample}
    t.shape = {1, #sample}

    local out = network:predict(t)
    local best_idx = 1
    local best_val = -math.huge
    for j, v in ipairs(out.data[1]) do
        if v > best_val then
            best_val = v
            best_idx = j
        end
    end
    return best_idx - 1
end

local correct = 0
for i = 1, #ds.test_data do
    local pred = predict_class(ds.test_data[i])
    if pred == ds.test_labels[i] then
        correct = correct + 1
    end
end

local acc = (correct / #ds.test_data) * 100
print(string.format("Test accuracy: %.2f%% (%d/%d)", acc, correct, #ds.test_data))

print("\nSample predictions (first 10):")
local preview = math.min(10, #ds.test_data)
for i = 1, preview do
    local pred = predict_class(ds.test_data[i])
    print(string.format("  %2d) predicted=%d actual=%d", i, pred, ds.test_labels[i]))
end
