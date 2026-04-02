local Tensor = require("core.tensor")

local MSELoss = {}
MSELoss.__index = MSELoss

function MSELoss.new()
    local self = setmetatable({}, MSELoss)
    self.prediction = nil
    self.target = nil
    return self
end

function MSELoss:forward(prediction, target)
    self.prediction = prediction
    self.target = target

    local batch_size = #prediction.data
    local feat_size = #prediction.data[1]
    local loss = 0

    for i = 1, batch_size do
        for j = 1, feat_size do
            local diff = prediction.data[i][j] - target.data[i][j]
            loss = loss + diff * diff
        end
    end

    return loss / (batch_size * feat_size)
end

function MSELoss:backward()
    local grad = Tensor.new()
    grad.data = {}
    grad.shape = self.prediction.shape

    local batch_size = #self.prediction.data
    local feat_size = #self.prediction.data[1]
    local scale = 2 / (batch_size * feat_size)

    for i = 1, batch_size do
        grad.data[i] = {}
        for j = 1, feat_size do
            grad.data[i][j] = (self.prediction.data[i][j] - self.target.data[i][j]) * scale
        end
    end

    return grad
end

return MSELoss
