-- core/tensor.lua
local Tensor = {}
Tensor.__index = Tensor

function Tensor.new(data)
    local self = setmetatable({}, Tensor)
    if data ~= nil then
        if type(data) == "table" then
            self.data = data
            self.shape = self:_get_shape(data)
        else
            self.data = {}
            self.shape = {}
        end
    else
        self.data = {}
        self.shape = {}
    end
    return self
end

function Tensor:_get_shape(t, dims)
    dims = dims or {}
    if type(t) == "table" then
        table.insert(dims, #t)
        if #t > 0 and type(t[1]) == "table" then
            return self:_get_shape(t[1], dims)
        end
    end
    return dims
end

function Tensor:zeros(shape)
    local function create_zeros(dims)
        if #dims == 1 then
            local t = {}
            for i = 1, dims[1] do 
                t[i] = 0 
            end
            return t
        elseif #dims == 2 then
            local t = {}
            for i = 1, dims[1] do
                t[i] = {}
                for j = 1, dims[2] do
                    t[i][j] = 0
                end
            end
            return t
        else
            local t = {}
            for i = 1, dims[1] do
                t[i] = create_zeros({table.unpack(dims, 2)})
            end
            return t
        end
    end
    self.data = create_zeros(shape)
    self.shape = shape
    return self
end

function Tensor:ones(shape)
    local function create_ones(dims)
        if #dims == 1 then
            local t = {}
            for i = 1, dims[1] do 
                t[i] = 1 
            end
            return t
        elseif #dims == 2 then
            local t = {}
            for i = 1, dims[1] do
                t[i] = {}
                for j = 1, dims[2] do
                    t[i][j] = 1
                end
            end
            return t
        else
            local t = {}
            for i = 1, dims[1] do
                t[i] = create_ones({table.unpack(dims, 2)})
            end
            return t
        end
    end
    self.data = create_ones(shape)
    self.shape = shape
    return self
end

function Tensor:random(shape, std)
    std = std or 0.1
    local function create_random(dims)
        if #dims == 1 then
            local t = {}
            for i = 1, dims[1] do 
                t[i] = (math.random() - 0.5) * 2 * std
            end
            return t
        elseif #dims == 2 then
            local t = {}
            for i = 1, dims[1] do
                t[i] = {}
                for j = 1, dims[2] do
                    t[i][j] = (math.random() - 0.5) * 2 * std
                end
            end
            return t
        else
            local t = {}
            for i = 1, dims[1] do
                t[i] = create_random({table.unpack(dims, 2)})
            end
            return t
        end
    end
    self.data = create_random(shape)
    self.shape = shape
    return self
end

function Tensor:T()
    -- Транспонирование для 2D матриц
    if #self.shape == 2 then
        local result = Tensor.new()
        result.shape = {self.shape[2], self.shape[1]}
        result.data = {}
        
        for i = 1, self.shape[2] do
            result.data[i] = {}
            for j = 1, self.shape[1] do
                if self.data[j] and self.data[j][i] ~= nil then
                    result.data[i][j] = self.data[j][i]
                else
                    result.data[i][j] = 0
                end
            end
        end
        return result
    end
    return self
end

function Tensor:matmul(other)
    -- Матричное умножение
    if not other or not other.data then
        error("matmul: other tensor is nil or has no data")
    end
    
    if #self.shape == 2 and #other.shape == 2 then
        local m, n = self.shape[1], self.shape[2]
        local p = other.shape[2]
        
        -- Проверка размерностей
        if n ~= other.shape[1] then
            error(string.format("Cannot matmul (%d x %d) with (%d x %d)", 
                  m, n, other.shape[1], p))
        end
        
        local result = Tensor.new()
        result.shape = {m, p}
        result.data = {}
        
        for i = 1, m do
            result.data[i] = {}
            for j = 1, p do
                local sum = 0
                for k = 1, n do
                    local a = self.data[i] and self.data[i][k] or 0
                    local b = other.data[k] and other.data[k][j] or 0
                    sum = sum + a * b
                end
                result.data[i][j] = sum
            end
        end
        return result
    end
    error("matmul only supports 2D tensors")
end

function Tensor:add(other)
    local result = Tensor.new()
    if type(other) == "number" then
        local function add_scalar(t)
            if type(t) == "table" then
                local res = {}
                for i, v in ipairs(t) do
                    res[i] = add_scalar(v)
                end
                return res
            else
                return t + other
            end
        end
        result.data = add_scalar(self.data)
    else
        local function add_tensor(t1, t2)
            if type(t1) == "table" and type(t2) == "table" then
                local res = {}
                for i = 1, #t1 do
                    res[i] = add_tensor(t1[i], t2[i])
                end
                return res
            else
                return t1 + t2
            end
        end
        result.data = add_tensor(self.data, other.data)
    end
    result.shape = self.shape
    return result
end

function Tensor:multiply(other)
    local result = Tensor.new()
    if type(other) == "number" then
        local function mul_scalar(t)
            if type(t) == "table" then
                local res = {}
                for i, v in ipairs(t) do
                    res[i] = mul_scalar(v)
                end
                return res
            else
                return t * other
            end
        end
        result.data = mul_scalar(self.data)
    else
        local function mul_tensor(t1, t2)
            if type(t1) == "table" and type(t2) == "table" then
                local res = {}
                for i = 1, #t1 do
                    res[i] = mul_tensor(t1[i], t2[i])
                end
                return res
            else
                return t1 * t2
            end
        end
        result.data = mul_tensor(self.data, other.data)
    end
    result.shape = self.shape
    return result
end

function Tensor:sum()
    local function sum_recursive(t)
        local s = 0
        if type(t) == "table" then
            for _, v in ipairs(t) do
                s = s + sum_recursive(v)
            end
        else
            s = t
        end
        return s
    end
    return sum_recursive(self.data)
end

function Tensor:clone()
    local function copy(t)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = copy(v)
            end
            return res
        else
            return t
        end
    end
    local result = Tensor.new()
    result.data = copy(self.data)
    result.shape = self.shape
    return result
end

function Tensor:__tostring()
    local function to_str(t, indent)
        indent = indent or 0
        if type(t) == "table" then
            if #t == 0 then return "{}" end
            local s = "{"
            for i, v in ipairs(t) do
                if i > 1 then s = s .. ", " end
                if type(v) == "table" then
                    s = s .. to_str(v, indent + 1)
                else
                    s = s .. string.format("%.4f", v)
                end
            end
            s = s .. "}"
            return s
        else
            return string.format("%.4f", t)
        end
    end
    return to_str(self.data)
end

return Tensor