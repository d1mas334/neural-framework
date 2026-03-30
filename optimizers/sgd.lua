-- optimizers/sgd.lua
local Tensor = require("core.tensor")

local SGD = {}
SGD.__index = SGD

function SGD.new(learning_rate)
    local self = setmetatable({}, SGD)
    self.lr = learning_rate or 0.01
    return self
end

function SGD:update(param, grad, param_name)
    local function update_recursive(p, g)
        if type(p) == "table" then
            local res = {}
            for i = 1, #p do
                res[i] = update_recursive(p[i], g[i])
            end
            return res
        else
            return p - self.lr * g
        end
    end
    
    local new_data = update_recursive(param.data, grad.data)
    
    local new_param = Tensor.new()
    new_param.data = new_data
    new_param.shape = param.shape
    return new_param
end

return SGD