-- core/network.lua
local Network = {}
Network.__index = Network

function Network.new(layers)
    local self = setmetatable({}, Network)
    self.layers = layers or {}
    return self
end

function Network:add_layer(layer)
    table.insert(self.layers, layer)
end

function Network:forward(input)
    local output = input
    for i, layer in ipairs(self.layers) do
        output = layer:forward(output)
    end
    return output
end

function Network:backward(grad_output)
    local grad = grad_output
    for i = #self.layers, 1, -1 do
        grad = self.layers[i]:backward(grad)
    end
    return grad
end

function Network:update_params(optimizer)
    for i, layer in ipairs(self.layers) do
        if layer.update_params then
            layer:update_params(optimizer, "layer_" .. i)
        end
    end
end

function Network:zero_grad()
    for _, layer in ipairs(self.layers) do
        if layer.zero_grad then
            layer:zero_grad()
        end
    end
end

function Network:train(data_loader, loss_fn, optimizer, epochs, verbose)
    verbose = (verbose == nil) and true or verbose
    
    for epoch = 1, epochs do
        local total_loss = 0
        local batch_count = 0
        
        data_loader:reset()
        
        for batch in data_loader:iter() do
            -- Forward pass
            local predictions = self:forward(batch.input)
            local loss = loss_fn:forward(predictions, batch.target)
            total_loss = total_loss + loss
            batch_count = batch_count + 1
            
            -- Backward pass
            local grad = loss_fn:backward()
            self:backward(grad)
            
            -- Update parameters
            self:update_params(optimizer)
            
            -- Zero gradients
            self:zero_grad()
        end
        
        local avg_loss = total_loss / batch_count
        if verbose then
            print(string.format("Epoch %d/%d, Loss: %.4f", epoch, epochs, avg_loss))
        end
    end
end

function Network:predict(input)
    return self:forward(input)
end

function Network:eval()
    for _, layer in ipairs(self.layers) do
        if layer.eval then
            layer:eval()
        end
    end
end

function Network:train_mode()
    for _, layer in ipairs(self.layers) do
        if layer.train then
            layer:train()
        end
    end
end

return Network