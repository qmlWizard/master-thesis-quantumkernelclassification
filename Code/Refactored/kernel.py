import pennylane as qml
from pennylane import numpy as np
from encoding import angle_encoding
from ansatz import efficient_su2

def variational_circuit(x, params, num_qubits, wires):
	for j, layer_params in enumerate(params):
		angle_encoding(
						input_data = x, 
				  		num_qubits = num_qubits, 
				  		wires = wires, 
						gate = 'RZ', 
						input_scaling = True, 
						input_scaling_param = layer_params[0, 0],  
						hadamard = True
					   )
		efficient_su2(num_qubits = num_qubits, params = layer_params[1:], wires = wires )

def kernel_circuit(x1, x2, num_qubits, wires, params):
	variational_circuit(
						x = x1,
					 	params = params[0],
						num_qubits = num_qubits,
						wires = wires
					   )
	
	adjoint_variational_circuit = qml.adjoint(variational_circuit)

	adjoint_variational_circuit(
						x = x2, 
						params = params [1],
						num_qubits = num_qubits,
						wires = wires
						)
		
	return qml.probs(wires=wires)

def kernel(x1, x2, params, wires, num_qubits):
	return kernel_circuit(x1, x2, params, wires, num_qubits)[0]
