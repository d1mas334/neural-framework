local GeneratedVsReal = {}

local function split_line(line, delimiter)
    local result = {}
    local pattern = "([^" .. delimiter .. "]+)"
    for value in string.gmatch(line, pattern) do
        result[#result + 1] = value
    end
    return result
end

local function parse_row(parts, label_col)
    local row = {}
    local label = nil

    for i = 1, #parts do
        local n = tonumber(parts[i])
        if not n then
            return nil, nil
        end

        if i == label_col then
            label = n
        else
            row[#row + 1] = n
        end
    end

    if label ~= 0 and label ~= 1 then
        return nil, nil
    end

    return row, label
end

local function shuffle_in_place(data, labels)
    for i = #data, 2, -1 do
        local j = math.random(i)
        data[i], data[j] = data[j], data[i]
        labels[i], labels[j] = labels[j], labels[i]
    end
end

function GeneratedVsReal.load(csv_path, opts)
    opts = opts or {}
    local delimiter = opts.delimiter or ","
    local has_header = (opts.has_header == nil) and true or opts.has_header
    local label_col = opts.label_col or 1
    local test_ratio = opts.test_ratio or 0.2
    local shuffle = (opts.shuffle == nil) and true or opts.shuffle
    local seed = opts.seed or 42

    local all_data = {}
    local all_labels = {}

    local line_idx = 0
    for line in io.lines(csv_path) do
        line_idx = line_idx + 1
        if line ~= "" and (not has_header or line_idx > 1) then
            local parts = split_line(line, delimiter)
            local row, label = parse_row(parts, label_col)
            if row and label ~= nil then
                all_data[#all_data + 1] = row
                all_labels[#all_labels + 1] = label
            end
        end
    end

    assert(#all_data > 0, "No valid rows found in CSV: " .. csv_path)
    assert(#all_data == #all_labels, "Data/label size mismatch")

    math.randomseed(seed)
    if shuffle then
        shuffle_in_place(all_data, all_labels)
    end

    local n = #all_data
    local test_size = math.max(1, math.floor(n * test_ratio))
    local train_size = n - test_size

    local train_data, train_labels = {}, {}
    local test_data, test_labels = {}, {}

    for i = 1, train_size do
        train_data[#train_data + 1] = all_data[i]
        train_labels[#train_labels + 1] = all_labels[i]
    end
    for i = train_size + 1, n do
        test_data[#test_data + 1] = all_data[i]
        test_labels[#test_labels + 1] = all_labels[i]
    end

    return {
        train_data = train_data,
        train_labels = train_labels,
        test_data = test_data,
        test_labels = test_labels,
        input_dim = #all_data[1]
    }
end

return GeneratedVsReal
