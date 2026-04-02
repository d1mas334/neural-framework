package.path = "../?.lua;" .. package.path

local Network = require("core.network")
local Linear = require("layers.linear")
local Activation = require("layers.activation")
local BinaryCrossEntropy = require("losses.binary_cross_entropy")
local Momentum = require("optimizers.momentum")
local Adam = require("optimizers.adam")
local DataLoader = require("core.dataloader")
local Tensor = require("core.tensor")
local Dataset = require("datasets.generated_vs_real")

local CSV_PATH = arg[1] or "../data/generated_vs_real.csv"
local EPOCHS = tonumber(arg[2]) or 10
local BATCH_SIZE = tonumber(arg[3]) or 64
local OPTIMIZER_NAME = arg[4] or "adam"

print("=" .. string.rep("=", 60))
print("Generated vs Real Image Classification (Binary)")
print("=" .. string.rep("=", 60))
print("Label convention: 0=real, 1=generated")
print("Input CSV format: label,p1,p2,...,pN")

local ds = Dataset.load(CSV_PATH, {
    delimiter = ",",
    has_header = true,
    label_col = 1,
    test_ratio = 0.2,
    shuffle = true,
    seed = 42
})

print(string.format("Dataset loaded: train=%d, test=%d, features=%d",
    #ds.train_data, #ds.test_data, ds.input_dim))

-- Split train -> train/val for threshold tuning (no extra training epochs)
local val_size = math.max(1, math.floor(#ds.train_data * 0.1))
local train_size = #ds.train_data - val_size

local train_data, train_labels = {}, {}
local val_data, val_labels = {}, {}

for i = 1, train_size do
    train_data[#train_data + 1] = ds.train_data[i]
    train_labels[#train_labels + 1] = ds.train_labels[i]
end
for i = train_size + 1, #ds.train_data do
    val_data[#val_data + 1] = ds.train_data[i]
    val_labels[#val_labels + 1] = ds.train_labels[i]
end

print(string.format("Train split: train=%d, val=%d, test=%d",
    #train_data, #val_data, #ds.test_data))

local function compute_feature_stats(data)
    local n = #data
    local d = #data[1]
    local mean, std = {}, {}

    for j = 1, d do
        mean[j] = 0
        std[j] = 0
    end

    for i = 1, n do
        local row = data[i]
        for j = 1, d do
            mean[j] = mean[j] + row[j]
        end
    end
    for j = 1, d do
        mean[j] = mean[j] / n
    end

    for i = 1, n do
        local row = data[i]
        for j = 1, d do
            local diff = row[j] - mean[j]
            std[j] = std[j] + diff * diff
        end
    end
    for j = 1, d do
        std[j] = math.sqrt(std[j] / n)
        if std[j] < 1e-6 then
            std[j] = 1
        end
    end

    return mean, std
end

local function standardize_in_place(data, mean, std)
    for i = 1, #data do
        local row = data[i]
        for j = 1, #row do
            row[j] = (row[j] - mean[j]) / std[j]
        end
    end
end

local mean, std = compute_feature_stats(train_data)
standardize_in_place(train_data, mean, std)
standardize_in_place(val_data, mean, std)
standardize_in_place(ds.test_data, mean, std)

local train_loader = DataLoader.new(train_data, train_labels, BATCH_SIZE, true)

local network = Network.new({
    Linear.new(ds.input_dim, 128),
    Activation.new("relu"),
    Linear.new(128, 32),
    Activation.new("relu"),
    Linear.new(32, 1),
    Activation.new("sigmoid")
})

local loss_fn = BinaryCrossEntropy.new()
local optimizer = nil
if OPTIMIZER_NAME == "momentum" then
    optimizer = Momentum.new(0.01, 0.9)
else
    optimizer = Adam.new(0.001, 0.9, 0.999, 1e-8)
    OPTIMIZER_NAME = "adam"
end

print(string.format("Training: epochs=%d, batch_size=%d, optimizer=%s", EPOCHS, BATCH_SIZE, OPTIMIZER_NAME))

local function copy_network_params(src_net)
    local snapshot = {}
    for li, layer in ipairs(src_net.layers) do
        if layer.params then
            snapshot[li] = {}
            for name, param in pairs(layer.params) do
                snapshot[li][name] = param:clone()
            end
        end
    end
    return snapshot
end

local function restore_network_params(dst_net, snapshot)
    for li, layer in ipairs(dst_net.layers) do
        if layer.params and snapshot[li] then
            for name, param in pairs(snapshot[li]) do
                layer.params[name] = param:clone()
            end
        end
    end
end

local function predict_class(sample, threshold)
    threshold = threshold or 0.5
    local t = Tensor.new()
    t.data = {sample}
    t.shape = {1, #sample}

    local out = network:predict(t)
    local p = out.data[1][1]
    if p >= threshold then
        return 1
    end
    return 0
end

local function compute_accuracy(data, labels, threshold)
    local correct = 0
    for i = 1, #data do
        local pred = predict_class(data[i], threshold)
        if pred == labels[i] then
            correct = correct + 1
        end
    end
    return (correct / #data) * 100, correct
end

local best_snapshot = nil
local best_epoch = 1
local best_epoch_val_acc = -1

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

    local val_acc = compute_accuracy(val_data, val_labels, 0.5)
    if val_acc > best_epoch_val_acc then
        best_epoch_val_acc = val_acc
        best_epoch = epoch
        best_snapshot = copy_network_params(network)
    end

    print(string.format("Epoch %d/%d | avg loss: %.6f | val@0.50: %.2f%%",
        epoch, EPOCHS, total_loss / batches, val_acc))
end

if best_snapshot then
    restore_network_params(network, best_snapshot)
end
print(string.format("Using best checkpoint from epoch %d (val@0.50: %.2f%%)", best_epoch, best_epoch_val_acc))

local best_threshold = 0.5
local best_val_acc = -1
for ti = 20, 80 do
    local th = ti / 100
    local acc = compute_accuracy(val_data, val_labels, th)
    if acc > best_val_acc then
        best_val_acc = acc
        best_threshold = th
    end
end

print(string.format("Best threshold on validation: %.2f (val acc: %.2f%%)", best_threshold, best_val_acc))

local test_acc, test_correct = compute_accuracy(ds.test_data, ds.test_labels, best_threshold)
print(string.format("Test accuracy: %.2f%% (%d/%d)", test_acc, test_correct, #ds.test_data))

print("\nSample predictions (first 10):")
local preview = math.min(10, #ds.test_data)
for i = 1, preview do
    local pred = predict_class(ds.test_data[i], best_threshold)
    local label = ds.test_labels[i]
    print(string.format("  %2d) predicted=%d actual=%d", i, pred, label))
end
