import enum
import mlx.core as mx
import mlx.nn as nn

# Enums
class MappingType(enum.Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(enum.Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

# LTCCell in MLX
class LTCCell(nn.Module):
    def __init__(self, num_units, input_mapping=MappingType.Affine,
                 solver=ODESolver.SemiImplicit, ode_solver_unfolds=6,
                 activation=mx.tanh, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._input_mapping = input_mapping
        self._solver = solver
        self._ode_solver_unfolds = ode_solver_unfolds
        self._activation = activation
        self.built = False

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        # input_shape: (..., input_dim)
        input_dim = input_shape[-1]
        self.kernel = self.add_parameter("kernel", (input_dim, self._num_units), initializer=nn.glorot_uniform)
        self.recurrent_kernel = self.add_parameter("recurrent_kernel", (self._num_units, self._num_units), initializer=nn.glorot_uniform)
        self.bias = self.add_parameter("bias", (self._num_units,), initializer=nn.zeros)
        self.built = True

    def __call__(self, inputs, states):
        if not self.built:
            self.build(inputs.shape)
        prev_output = states[0]
        net_input = mx.matmul(inputs, self.kernel)
        net_input += mx.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias

        if self._solver == ODESolver.SemiImplicit:
            output = self._semi_implicit_solver(prev_output, net_input)
        elif self._solver == ODESolver.Explicit:
            output = self._explicit_solver(prev_output, net_input)
        elif self._solver == ODESolver.RungeKutta:
            output = self._runge_kutta_solver(prev_output, net_input)
        else:
            raise ValueError("Unsupported ODE Solver type.")

        return output, [output]

    def _semi_implicit_solver(self, prev_output, net_input):
        # Euler-style update toward activation result.
        return prev_output + self._ode_solver_unfolds * (self._activation(net_input) - prev_output)

    def _explicit_solver(self, prev_output, net_input):
        return prev_output + self._ode_solver_unfolds * self._activation(net_input)

    def _runge_kutta_solver(self, prev_output, net_input):
        dt = 1.0 / self._ode_solver_unfolds
        k1 = self._activation(net_input)
        k2 = self._activation(net_input + 0.5 * dt * k1)
        k3 = self._activation(net_input + 0.5 * dt * k2)
        k4 = self._activation(net_input + dt * k3)
        return prev_output + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_units": self._num_units,
            "solver": self._solver,
            "ode_solver_unfolds": self._ode_solver_unfolds
        })
        return config

# CTRNN in MLX
class CTRNN(nn.Module):
    def __init__(self, units, global_feedback=False, activation=mx.tanh, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.global_feedback = global_feedback
        self.activation = activation
        self.cell_clip = cell_clip
        self.built = False

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_parameter("kernel", (input_dim, self.units), initializer=nn.glorot_uniform)
        self.recurrent_kernel = self.add_parameter("recurrent_kernel", (self.units, self.units), initializer=nn.glorot_uniform)
        self.bias = self.add_parameter("bias", (self.units,), initializer=nn.zeros)
        self.built = True

    def __call__(self, inputs, states):
        if not self.built:
            self.build(inputs.shape)
        prev_output = states[0]
        net_input = mx.matmul(inputs, self.kernel)
        net_input += mx.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = self.activation(net_input)
        if self.cell_clip is not None:
            output = mx.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]

# NODE in MLX
class NODE(nn.Module):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip
        self.built = False

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_parameter("kernel", (input_dim, self.units), initializer=nn.glorot_uniform)
        self.recurrent_kernel = self.add_parameter("recurrent_kernel", (self.units, self.units), initializer=nn.glorot_uniform)
        self.bias = self.add_parameter("bias", (self.units,), initializer=nn.zeros)
        self.built = True

    def __call__(self, inputs, states):
        if not self.built:
            self.build(inputs.shape)
        prev_output = states[0]
        net_input = mx.matmul(inputs, self.kernel)
        net_input += mx.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = mx.tanh(net_input)
        if self.cell_clip is not None:
            output = mx.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]

# CTGRU in MLX
class CTGRU(nn.Module):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip
        self.built = False

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Combined kernel for z and r gates.
        self.kernel = self.add_parameter("kernel", (input_dim, 2 * self.units), initializer=nn.glorot_uniform)
        self.recurrent_kernel = self.add_parameter("recurrent_kernel", (self.units, 2 * self.units), initializer=nn.glorot_uniform)
        self.bias = self.add_parameter("bias", (2 * self.units,), initializer=nn.zeros)
        # Parameters for candidate c.
        self.kernel_c = self.add_parameter("kernel_c", (input_dim, self.units), initializer=nn.glorot_uniform)
        self.recurrent_kernel_c = self.add_parameter("recurrent_kernel_c", (self.units, self.units), initializer=nn.glorot_uniform)
        self.bias_c = self.add_parameter("bias_c", (self.units,), initializer=nn.zeros)
        self.built = True

    def __call__(self, inputs, states):
        if not self.built:
            self.build(inputs.shape)
        prev_output = states[0]
        zr = mx.matmul(inputs, self.kernel)
        zr += mx.matmul(prev_output, self.recurrent_kernel)
        zr += self.bias
        z, r = mx.split(zr, 2, axis=-1)
        z = mx.sigmoid(z)
        r = mx.sigmoid(r)
        c = mx.matmul(inputs, self.kernel_c)
        c += r * mx.matmul(prev_output, self.recurrent_kernel_c)
        c += self.bias_c
        c = mx.tanh(c)
        output = (1 - z) * prev_output + z * c
        if self.cell_clip is not None:
            output = mx.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]