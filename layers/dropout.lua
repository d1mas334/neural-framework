-- layers/dropout.lua
local Dropout = {}
Dropout.__index = Dropout

function Dropout.new(dropout_rate)
    local self = setmetatable({}, Dropout)
    self.dropout_rate = dropout_rate or 0.5
    self.mask = nil
    self.training = true
    return self
end

function Dropout:forward(input)
    if not self.training then
        return input
    end
    
    local mask_data = {}
    local output_data = {}
    
    local function create_mask_and_apply(t)
        if type(t) == "table" then
            local mask_row = {}
            local output_row = {}
            for i, v in ipairs(t) do
                local m, o = create_mask_and_apply(v)
                mask_row[i] = m
                output_row[i] = o
            end
            return mask_row, output_row
        else
            local keep_prob = 1 - self.dropout_rate
            local mask = math.random() < keep_prob and 1 or 0
            local output = t * mask / keep_prob
            return mask, output
        end
    end
    
    self.mask, output_data = create_mask_and_apply(input.data)
    
    local result = require("core.tensor").new()
    result.data = output_data
    result.shape = input.shape
    return result
end

function Dropout:backward(grad_output)
    if not self.training then
        return grad_output
    end
    
    local grad_input_data = {}
    
    local function apply_mask(grad, mask)
        if type(grad) == "table" then
            local res = {}
            for i, v in ipairs(grad) do
                res[i] = apply_mask(v, mask[i])
            end
            return res
        else
            local keep_prob = 1 - self.dropout_rate
            return grad * mask / keep_prob
        end
    end
    
    grad_input_data = apply_mask(grad_output.data, self.mask)
    
    local result = require("core.tensor").new()
    result.data = grad_input_data
    result.shape = grad_output.shape
    return result
end

function Dropout:train()
    self.training = true
end

function Dropout:eval()
    self.training = false
end

function Dropout:update_params(optimizer, layer_name)
    -- Dropout не имеет параметров
end

function Dropout:zero_grad()
    -- Dropout не имеет градиентов
end

return Dropout