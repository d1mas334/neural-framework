-- losses/cross_entropy.lua
local CrossEntropyLoss = {}
CrossEntropyLoss.__index = CrossEntropyLoss

function CrossEntropyLoss.new()
    local self = setmetatable({}, CrossEntropyLoss)
    self.prediction = nil
    self.target = nil
    return self
end

function CrossEntropyLoss:forward(prediction, target)
    self.prediction = prediction
    self.target = target
    
    local loss = 0
    local batch_size = #prediction.data
    
    for i = 1, batch_size do
        local pred_row = prediction.data[i]
        local target_idx = target.data[i][1] + 1 -- Индексация с 0
        
        -- Вычисляем логистическую потерю
        local p = math.max(pred_row[target_idx], 1e-7)
        p = math.min(p, 1 - 1e-7)
        loss = loss - math.log(p)
    end
    
    return loss / batch_size
end

function CrossEntropyLoss:backward()
    local grad = require("core.tensor").new()
    grad.data = {}
    grad.shape = self.prediction.shape
    
    local batch_size = #self.prediction.data
    
    for i = 1, batch_size do
        grad.data[i] = {}
        for j = 1, #self.prediction.data[i] do
            grad.data[i][j] = self.prediction.data[i][j]
        end
        local target_idx = self.target.data[i][1] + 1
        grad.data[i][target_idx] = grad.data[i][target_idx] - 1
    end
    
    -- Нормируем на размер батча
    for i = 1, batch_size do
        for j = 1, #grad.data[i] do
            grad.data[i][j] = grad.data[i][j] / batch_size
        end
    end
    
    return grad
end

return CrossEntropyLoss