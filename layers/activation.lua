-- layers/activation.lua
local Activation = {}
Activation.__index = Activation

function Activation.new(activation_type)
    local self = setmetatable({}, Activation)
    self.activation_type = activation_type
    self.input = nil
    return self
end

function Activation:forward(input)
    self.input = input
    
    local output_data = nil
    
    if self.activation_type == "relu" then
        output_data = self:_relu_forward(input.data)
    elseif self.activation_type == "sigmoid" then
        output_data = self:_sigmoid_forward(input.data)
    elseif self.activation_type == "tanh" then
        output_data = self:_tanh_forward(input.data)
    elseif self.activation_type == "softmax" then
        output_data = self:_softmax_forward(input.data)
    elseif self.activation_type == "leaky_relu" then
        output_data = self:_leaky_relu_forward(input.data)
    else
        error("Unknown activation type: " .. self.activation_type)
    end
    
    local result = require("core.tensor").new()
    result.data = output_data
    result.shape = input.shape
    return result
end

function Activation:backward(grad_output)
    local grad_input_data = nil
    
    if self.activation_type == "relu" then
        grad_input_data = self:_relu_backward(grad_output.data, self.input.data)
    elseif self.activation_type == "sigmoid" then
        grad_input_data = self:_sigmoid_backward(grad_output.data, self.input.data)
    elseif self.activation_type == "tanh" then
        grad_input_data = self:_tanh_backward(grad_output.data, self.input.data)
    elseif self.activation_type == "softmax" then
        grad_input_data = self:_softmax_backward(grad_output.data, self.input.data)
    elseif self.activation_type == "leaky_relu" then
        grad_input_data = self:_leaky_relu_backward(grad_output.data, self.input.data)
    else
        error("Unknown activation type: " .. self.activation_type)
    end
    
    local result = require("core.tensor").new()
    result.data = grad_input_data
    result.shape = self.input.shape
    return result
end

-- ReLU
function Activation:_relu_forward(x)
    local function relu(t)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = relu(v)
            end
            return res
        else
            return math.max(0, t)
        end
    end
    return relu(x)
end

function Activation:_relu_backward(grad_out, x)
    local function relu_grad(t, g)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = relu_grad(v, g[i])
            end
            return res
        else
            return t > 0 and g or 0
        end
    end
    return relu_grad(x, grad_out)
end

-- Leaky ReLU
function Activation:_leaky_relu_forward(x, alpha)
    alpha = alpha or 0.01
    local function leaky_relu(t)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = leaky_relu(v)
            end
            return res
        else
            return t > 0 and t or alpha * t
        end
    end
    return leaky_relu(x)
end

function Activation:_leaky_relu_backward(grad_out, x, alpha)
    alpha = alpha or 0.01
    local function leaky_relu_grad(t, g)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = leaky_relu_grad(v, g[i])
            end
            return res
        else
            return g * (t > 0 and 1 or alpha)
        end
    end
    return leaky_relu_grad(x, grad_out)
end

-- Sigmoid
function Activation:_sigmoid_forward(x)
    local function sigmoid(t)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = sigmoid(v)
            end
            return res
        else
            return 1 / (1 + math.exp(-t))
        end
    end
    return sigmoid(x)
end

function Activation:_sigmoid_backward(grad_out, x)
    local function sigmoid_grad(t, g)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = sigmoid_grad(v, g[i])
            end
            return res
        else
            local s = 1 / (1 + math.exp(-t))
            return g * s * (1 - s)
        end
    end
    return sigmoid_grad(x, grad_out)
end

-- Tanh
function Activation:_tanh_forward(x)
    local function tanh(t)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = tanh(v)
            end
            return res
        else
            return math.tanh(t)
        end
    end
    return tanh(x)
end

function Activation:_tanh_backward(grad_out, x)
    local function tanh_grad(t, g)
        if type(t) == "table" then
            local res = {}
            for i, v in ipairs(t) do
                res[i] = tanh_grad(v, g[i])
            end
            return res
        else
            local th = math.tanh(t)
            return g * (1 - th * th)
        end
    end
    return tanh_grad(x, grad_out)
end

-- Softmax
function Activation:_softmax_forward(x)
    local result = {}
    
    for i = 1, #x do
        result[i] = {}
        
        -- Находим максимальное значение для численной стабильности
        local max_val = -math.huge
        for j = 1, #x[i] do
            if x[i][j] > max_val then
                max_val = x[i][j]
            end
        end
        
        -- Вычисляем экспоненты
        local sum = 0
        for j = 1, #x[i] do
            result[i][j] = math.exp(x[i][j] - max_val)
            sum = sum + result[i][j]
        end
        
        -- Нормализуем
        for j = 1, #x[i] do
            result[i][j] = result[i][j] / sum
        end
    end
    
    return result
end

function Activation:_softmax_backward(grad_out, x)
    -- Для softmax с cross-entropy loss градиент упрощается
    -- Здесь реализуем общий случай
    local grad_input = {}
    
    for i = 1, #grad_out do
        grad_input[i] = {}
        local softmax_row = {}
        
        -- Вычисляем softmax значения
        local max_val = -math.huge
        for j = 1, #x[i] do
            if x[i][j] > max_val then
                max_val = x[i][j]
            end
        end
        
        local sum = 0
        for j = 1, #x[i] do
            softmax_row[j] = math.exp(x[i][j] - max_val)
            sum = sum + softmax_row[j]
        end
        
        for j = 1, #x[i] do
            softmax_row[j] = softmax_row[j] / sum
        end
        
        -- Вычисляем градиент: S * (I - S^T)
        for j = 1, #grad_out[i] do
            local grad_sum = 0
            for k = 1, #grad_out[i] do
                if j == k then
                    grad_sum = grad_sum + grad_out[i][k] * softmax_row[j] * (1 - softmax_row[j])
                else
                    grad_sum = grad_sum - grad_out[i][k] * softmax_row[j] * softmax_row[k]
                end
            end
            grad_input[i][j] = grad_sum
        end
    end
    
    return grad_input
end

function Activation:update_params(optimizer, layer_name)
    -- Активационные слои не имеют обучаемых параметров
end

function Activation:zero_grad()
    -- Активационные слои не имеют градиентов
end

return Activation