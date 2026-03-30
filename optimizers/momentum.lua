-- optimizers/momentum.lua
local Tensor = require("core.tensor")

local Momentum = {}
Momentum.__index = Momentum

function Momentum.new(learning_rate, momentum)
    local self = setmetatable({}, Momentum)
    self.lr = learning_rate or 0.01
    self.momentum = momentum or 0.9
    self.velocity = {}
    return self
end

function Momentum:update(param, grad, param_name)
    if not self.velocity[param_name] then
        -- Инициализируем velocity нулями
        local vel = Tensor.new()
        vel:zeros(param.shape)
        self.velocity[param_name] = vel
    end
    
    local function update_recursive(p, g, v)
        if type(p) == "table" then
            local res = {}
            local new_v = {}
            for i = 1, #p do
                res[i], new_v[i] = update_recursive(p[i], g[i], v[i])
            end
            return res, new_v
        else
            local new_v = self.momentum * v - self.lr * g
            local new_p = p + new_v
            return new_p, new_v
        end
    end
    
    local new_data, new_velocity = update_recursive(
        param.data, 
        grad.data, 
        self.velocity[param_name].data
    )
    
    self.velocity[param_name].data = new_velocity
    
    local new_param = Tensor.new()
    new_param.data = new_data
    new_param.shape = param.shape
    return new_param
end

return Momentum