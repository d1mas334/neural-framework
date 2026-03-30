-- core/layer.lua
local Layer = {}
Layer.__index = Layer

function Layer.new()
    local self = setmetatable({}, Layer)
    self.params = {}
    self.grads = {}
    return self
end

function Layer:forward(input)
    error("forward method must be implemented")
end

function Layer:backward(grad_output)
    error("backward method must be implemented")
end

function Layer:update_params(optimizer)
    for name, param in pairs(self.params) do
        local grad = self.grads[name]
        if grad then
            self.params[name] = optimizer:update(param, grad, name)
        end
    end
end

return Layer