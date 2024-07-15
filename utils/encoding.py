import pennylane as qml
from pennylane import numpy as np
from jax import numpy as jnp
from autograd.numpy import array as ag_array
from autograd.numpy.numpy_boxes import ArrayBox


def angle_encoding(input_data, num_qubits, wires, gate = 'RZ', input_scaling_param = 1, input_scaling = False, hadamard = True):

	if isinstance(input_scaling_param, ArrayBox):
		input_scaling_param = jnp.array(input_scaling_param._value)

	if input_scaling:
		input_data = input_data * jnp.array(input_scaling_param)

	for i, wire in enumerate(wires):
		if hadamard:
			qml.Hadamard(wires = [wire])
		
		if gate == 'RZ':
			qml.RZ(input_data[i], wires=[wire])
		elif gate == 'RX':
			qml.RX(input_data[i], wires=[wire])
		elif gate == 'RY':
			qml.RY(input_data[i], wires=[wire])