local Tensor = require("core.tensor")

local BinaryCrossEntropy = {}
BinaryCrossEntropy.__index = BinaryCrossEntropy

function BinaryCrossEntropy.new()
    local self = setmetatable({}, BinaryCrossEntropy)
    self.prediction = nil
    self.target = nil
    self.eps = 1e-7
    return self
end

function BinaryCrossEntropy:forward(prediction, target)
    self.prediction = prediction
    self.target = target

    local loss = 0
    local batch_size = #prediction.data

    for i = 1, batch_size do
        local p = prediction.data[i][1]
        local y = target.data[i][1]

        if p < self.eps then p = self.eps end
        if p > 1 - self.eps then p = 1 - self.eps end

        loss = loss - (y * math.log(p) + (1 - y) * math.log(1 - p))
    end

    return loss / batch_size
end

function BinaryCrossEntropy:backward()
    local grad = Tensor.new()
    grad.data = {}
    grad.shape = self.prediction.shape

    local batch_size = #self.prediction.data

    for i = 1, batch_size do
        local p = self.prediction.data[i][1]
        local y = self.target.data[i][1]

        if p < self.eps then p = self.eps end
        if p > 1 - self.eps then p = 1 - self.eps end

        local d = -((y / p) - ((1 - y) / (1 - p)))
        grad.data[i] = {d / batch_size}
    end

    return grad
end

return BinaryCrossEntropy
