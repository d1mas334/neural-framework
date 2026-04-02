-- datasets/mnist.lua
local MNIST = {}

local function read_all_bytes(path)
    local f = assert(io.open(path, "rb"), "Cannot open file: " .. path)
    local data = f:read("*all")
    f:close()
    return data
end

local function read_u32_be(bytes, offset)
    local b1, b2, b3, b4 = bytes:byte(offset, offset + 3)
    return ((b1 * 256 + b2) * 256 + b3) * 256 + b4
end

local function parse_images(path, limit, normalize)
    local bytes = read_all_bytes(path)

    local magic = read_u32_be(bytes, 1)
    assert(magic == 2051, "Invalid image file magic in " .. path)

    local count = read_u32_be(bytes, 5)
    local rows = read_u32_be(bytes, 9)
    local cols = read_u32_be(bytes, 13)

    local image_size = rows * cols
    local n = math.min(limit or count, count)

    local images = {}
    local offset = 17

    for i = 1, n do
        local sample = {}
        for j = 1, image_size do
            local v = bytes:byte(offset)
            offset = offset + 1
            if normalize then
                sample[j] = v / 255.0
            else
                sample[j] = v
            end
        end
        images[i] = sample
    end

    return images, count, rows, cols
end

local function parse_labels(path, limit)
    local bytes = read_all_bytes(path)

    local magic = read_u32_be(bytes, 1)
    assert(magic == 2049, "Invalid label file magic in " .. path)

    local count = read_u32_be(bytes, 5)
    local n = math.min(limit or count, count)

    local labels = {}
    local offset = 9

    for i = 1, n do
        labels[i] = bytes:byte(offset)
        offset = offset + 1
    end

    return labels, count
end

function MNIST.load(data_path, opts)
    opts = opts or {}

    local normalize = (opts.normalize == nil) and true or opts.normalize
    local train_limit = opts.train_limit
    local test_limit = opts.test_limit

    local base = data_path or "data/mnist"
    local train_images_path = base .. "/train-images-idx3-ubyte"
    local train_labels_path = base .. "/train-labels-idx1-ubyte"
    local test_images_path = base .. "/t10k-images-idx3-ubyte"
    local test_labels_path = base .. "/t10k-labels-idx1-ubyte"

    print("Loading real MNIST from " .. base)

    local train_data, train_total, rows, cols = parse_images(train_images_path, train_limit, normalize)
    local train_labels, train_labels_total = parse_labels(train_labels_path, train_limit)

    local test_data, test_total = parse_images(test_images_path, test_limit, normalize)
    local test_labels, test_labels_total = parse_labels(test_labels_path, test_limit)

    assert(#train_data == #train_labels, "Train images/labels count mismatch")
    assert(#test_data == #test_labels, "Test images/labels count mismatch")

    print(string.format("MNIST loaded: train %d/%d, test %d/%d, image %dx%d", #train_data, train_total, #test_data, test_total, rows, cols))

    return {
        train_data = train_data,
        train_labels = train_labels,
        test_data = test_data,
        test_labels = test_labels,
        meta = {
            rows = rows,
            cols = cols,
            train_total = train_total,
            train_labels_total = train_labels_total,
            test_total = test_total,
            test_labels_total = test_labels_total,
            normalized = normalize
        }
    }
end

return MNIST