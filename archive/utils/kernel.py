import pennylane as qml
import numpy as np
from utils.encoding import angle_encoding
from utils.ansatz import efficient_su2

def variational_circuit(x, params, num_qubits, wires):

	layers = len(params) - 1
	
	for j in range(layers):
		angle_encoding(
						input_data = x, 
				  		num_qubits = num_qubits, 
				  		wires = wires, 
						gate = 'RZ', 
						input_scaling = False, 
						input_scaling_param = params[j, 0, 0, 0],  
						hadamard = True
					   )
		
		efficient_su2(num_qubits = num_qubits, params = params[j, 1:], wires = wires )

def kernel_circuit(x1, x2, num_qubits, wires, params = np.asarray([])):
	
	variational_circuit(
						x = x1,
					 	params = params[0],
						num_qubits = num_qubits,
						wires = wires
					   )
	
	adjoint_variational_circuit = qml.adjoint(variational_circuit)

	adjoint_variational_circuit(
						x = x2, 
						params = params[1],
						num_qubits = num_qubits,
						wires = wires
						)
		
	return qml.probs(wires=wires)

def kernel_without_ansatz(x1, x2, wires, num_qubits):
	angle_encoding(
						input_data = x1, 
				  		num_qubits = num_qubits, 
				  		wires = wires, 
						gate = 'RZ', 
						hadamard = True
					   )
	
	adjoint_angle_encoding = qml.adjoint(variational_circuit)

	adjoint_angle_encoding(
						x = x2, 
						num_qubits = num_qubits,
						gate = 'RZ',
						wires = wires
						)
