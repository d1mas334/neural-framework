-- core/dataloader.lua
local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.new(data, targets, batch_size, shuffle)
    local self = setmetatable({}, DataLoader)
    self.data = data
    self.targets = targets
    self.batch_size = batch_size or 32
    self.shuffle = shuffle or false
    self.indices = {}
    
    for i = 1, #data do
        table.insert(self.indices, i)
    end
    
    if self.shuffle then
        self:_shuffle_indices()
    end
    
    self.current_idx = 1
    return self
end

function DataLoader:_shuffle_indices()
    for i = #self.indices, 2, -1 do
        local j = math.random(i)
        self.indices[i], self.indices[j] = self.indices[j], self.indices[i]
    end
end

function DataLoader:reset()
    self.current_idx = 1
    if self.shuffle then
        self:_shuffle_indices()
    end
end

function DataLoader:iter()
    local self_ref = self
    return function()
        if self_ref.current_idx > #self_ref.indices then
            return nil
        end
        
        local batch_indices = {}
        for i = 1, self_ref.batch_size do
            if self_ref.current_idx <= #self_ref.indices then
                table.insert(batch_indices, self_ref.indices[self_ref.current_idx])
                self_ref.current_idx = self_ref.current_idx + 1
            end
        end
        
        local batch_input = require("core.tensor").new()
        local batch_target = require("core.tensor").new()
        
        batch_input.data = {}
        batch_target.data = {}
        
        for _, idx in ipairs(batch_indices) do
            table.insert(batch_input.data, self_ref.data[idx])
            table.insert(batch_target.data, {self_ref.targets[idx]})
        end
        
        batch_input.shape = {#batch_input.data, #self_ref.data[1]}
        batch_target.shape = {#batch_target.data, 1}
        
        return {input = batch_input, target = batch_target}
    end
end

function DataLoader:map(func)
    -- Применяет функцию ко всем данным
    for i = 1, #self.data do
        self.data[i] = func(self.data[i])
    end
    return self
end

return DataLoader