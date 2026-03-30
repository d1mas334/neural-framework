-- layers/linear.lua
local Tensor = require("core.tensor")

local Linear = {}
Linear.__index = Linear

function Linear.new(input_size, output_size)
    local self = setmetatable({}, Linear)
    
    -- Инициализация параметров
    self.params = {}
    self.grads = {}
    
    -- Веса (инициализация Xavier)
    local scale = math.sqrt(2.0 / input_size)
    self.params.W = Tensor.new()
    self.params.W:random({input_size, output_size}, scale)
    
    -- Смещения (bias)
    self.params.b = Tensor.new()
    self.params.b:zeros({1, output_size})
    
    -- Градиенты
    self.grads.W = Tensor.new()
    self.grads.W:zeros({input_size, output_size})
    self.grads.b = Tensor.new()
    self.grads.b:zeros({1, output_size})
    
    self.input = nil
    
    return self
end

function Linear:forward(input)
    self.input = input
    
    -- Проверка размерностей
    if #input.shape ~= 2 then
        error("Linear layer expects 2D input (batch_size, features)")
    end
    
    -- output = input * W + b
    local output = input:matmul(self.params.W)
    
    -- Добавляем смещение к каждой строке
    for i = 1, #output.data do
        for j = 1, #output.data[i] do
            output.data[i][j] = output.data[i][j] + self.params.b.data[1][j]
        end
    end
    
    return output
end

function Linear:backward(grad_output)
    -- Проверка наличия input
    if not self.input then
        error("Backward called without forward pass")
    end
    
    -- Получаем транспонированную версию входа
    local input_T = self.input:T()
    
    -- Градиент по весам: input.T * grad_output
    self.grads.W = input_T:matmul(grad_output)
    
    -- Градиент по смещению: сумма grad_output по батчу
    local batch_size = #grad_output.data
    if batch_size > 0 and #self.grads.b.data > 0 and #self.grads.b.data[1] > 0 then
        for j = 1, #self.grads.b.data[1] do
            local sum = 0
            for i = 1, batch_size do
                if grad_output.data[i] and grad_output.data[i][j] then
                    sum = sum + grad_output.data[i][j]
                end
            end
            self.grads.b.data[1][j] = sum
        end
    end
    
    -- Получаем транспонированную версию весов
    local W_T = self.params.W:T()
    
    -- Градиент по входу: grad_output * W.T
    local grad_input = grad_output:matmul(W_T)
    return grad_input
end

function Linear:update_params(optimizer, layer_name)
    for name, param in pairs(self.params) do
        local grad = self.grads[name]
        if grad then
            local full_name = layer_name and (layer_name .. "." .. name) or name
            local new_param = optimizer:update(param, grad, full_name)
            self.params[name] = new_param
        end
    end
end

function Linear:zero_grad()
    -- Обнуление градиентов
    local function zeros_recursive(t)
        if type(t) == "table" then
            for i = 1, #t do
                if type(t[i]) == "table" then
                    zeros_recursive(t[i])
                else
                    t[i] = 0
                end
            end
        end
    end
    
    if self.grads.W and self.grads.W.data then
        zeros_recursive(self.grads.W.data)
    end
    if self.grads.b and self.grads.b.data then
        zeros_recursive(self.grads.b.data)
    end
end

return Linear