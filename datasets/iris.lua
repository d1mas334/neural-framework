local Iris = {}

local function shuffle_in_place(data, labels)
    for i = #data, 2, -1 do
        local j = math.random(i)
        data[i], data[j] = data[j], data[i]
        labels[i], labels[j] = labels[j], labels[i]
    end
end

-- Synthetic but iris-like dataset:
-- 4 features with class-specific centers close to classic Iris ranges.
local function generate_iris_like(seed)
    math.randomseed(seed or 42)

    local centers = {
        {5.0, 3.4, 1.5, 0.2}, -- setosa
        {5.9, 2.8, 4.3, 1.3}, -- versicolor
        {6.5, 3.0, 5.5, 2.0}  -- virginica
    }
    local std = {0.35, 0.30, 0.45, 0.25}

    local data, labels = {}, {}
    local per_class = 50

    for cls = 1, 3 do
        for _ = 1, per_class do
            local sample = {}
            for f = 1, 4 do
                local noise = (math.random() * 2 - 1) * std[f]
                sample[f] = centers[cls][f] + noise
            end
            data[#data + 1] = sample
            labels[#labels + 1] = cls - 1
        end
    end

    return data, labels
end

function Iris.load(opts)
    opts = opts or {}
    local test_ratio = opts.test_ratio or 0.2
    local shuffle = (opts.shuffle == nil) and true or opts.shuffle
    local seed = opts.seed or 42

    local data, labels = generate_iris_like(seed)
    if shuffle then
        shuffle_in_place(data, labels)
    end

    local total = #data
    local test_size = math.max(1, math.floor(total * test_ratio))
    local train_size = total - test_size

    local train_data, train_labels = {}, {}
    local test_data, test_labels = {}, {}

    for i = 1, train_size do
        train_data[#train_data + 1] = data[i]
        train_labels[#train_labels + 1] = labels[i]
    end
    for i = train_size + 1, total do
        test_data[#test_data + 1] = data[i]
        test_labels[#test_labels + 1] = labels[i]
    end

    return {
        train_data = train_data,
        train_labels = train_labels,
        test_data = test_data,
        test_labels = test_labels,
        input_dim = 4,
        classes = 3
    }
end

return Iris
