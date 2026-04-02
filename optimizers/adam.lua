local Tensor = require("core.tensor")

local Adam = {}
Adam.__index = Adam

function Adam.new(learning_rate, beta1, beta2, eps)
    local self = setmetatable({}, Adam)
    self.lr = learning_rate or 0.001
    self.beta1 = beta1 or 0.9
    self.beta2 = beta2 or 0.999
    self.eps = eps or 1e-8
    self.m = {}
    self.v = {}
    self.t = {}
    return self
end

local function zeros_like(shape)
    local t = Tensor.new()
    t:zeros(shape)
    return t
end

function Adam:update(param, grad, param_name)
    param_name = param_name or "param"

    if not self.m[param_name] then
        self.m[param_name] = zeros_like(param.shape)
        self.v[param_name] = zeros_like(param.shape)
        self.t[param_name] = 0
    end

    self.t[param_name] = self.t[param_name] + 1
    local step = self.t[param_name]

    local beta1_pow = self.beta1 ^ step
    local beta2_pow = self.beta2 ^ step

    local function update_recursive(p, g, m, v)
        if type(p) == "table" then
            local out = {}
            local new_m = {}
            local new_v = {}
            for i = 1, #p do
                out[i], new_m[i], new_v[i] = update_recursive(p[i], g[i], m[i], v[i])
            end
            return out, new_m, new_v
        end

        local m_t = self.beta1 * m + (1 - self.beta1) * g
        local v_t = self.beta2 * v + (1 - self.beta2) * g * g

        local m_hat = m_t / (1 - beta1_pow)
        local v_hat = v_t / (1 - beta2_pow)

        local p_t = p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return p_t, m_t, v_t
    end

    local new_data, new_m, new_v = update_recursive(
        param.data,
        grad.data,
        self.m[param_name].data,
        self.v[param_name].data
    )

    self.m[param_name].data = new_m
    self.v[param_name].data = new_v

    local new_param = Tensor.new()
    new_param.data = new_data
    new_param.shape = param.shape
    return new_param
end

return Adam
