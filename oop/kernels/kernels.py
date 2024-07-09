import pennylane as qml
from pennylane import numpy as np
import concurrent.futures
from itertools import product
from sklearn.svm import SVC
import os
import time

dev = qml.device("default.qubit", wires=3, shots=None)
wires = dev.wires.tolist()

class kernel:

    def __init__(self, encoding_type, ansatz_type, num_qubits, num_wires):
        self.encoding_type
        self.ansatz_type
        self.num_qubits
        self.num_wires

    def encoding(self, input_data, wires, gate = 'RZ', input_scaling_param = 1, input_scaling = False, hadamard = True):
        
        if self.encoding_type == 'angle':
            if input_scaling:
                input_data = input_data * input_scaling_param
            
            for i, wire in enumerate(wires):
                if hadamard:
                    qml.Hadamard(wires = [wire])
                
                if gate == 'RZ':
                    qml.RZ(input_data[i], wires=[wire])
                elif gate == 'RX':
                    qml.RX(input_data[i], wires=[wire])
                elif gate == 'RY':
                    qml.RY(input_data[i], wires=[wire])
        
    def ansatz(self, num_qubits, params, wires):
        
        if self.ansatz_type == 'efficient_su2':
            for i in range(num_qubits):
                qml.RX(params[0, i, 0], wires=wires[i])
                qml.RY(params[0, i, 1], wires=wires[i])

            for i in range(num_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            
            qml.CNOT(wires=[wires[num_qubits - 1], wires[0]])
            
            for i in range(num_qubits):
                qml.RX(params[0, i, 2], wires=wires[i])
                qml.RY(params[0, i, 3], wires=wires[i])

    def random_parameters(self):
        pass
    
    def sampling(self):
        pass

    def uncertinity_sampling(self):
        pass

    def variational_circuit(self, x, params, num_qubits, wires):
        layers = len(params) - 1
	
        for j in range(layers):
            
            self.encoding(
                            input_data = x, 
                            num_qubits = num_qubits, 
                            wires = wires, 
                            gate = 'RZ', 
                            input_scaling = True, 
                            input_scaling_param = params[j, 0, 0, 0],  
                            hadamard = True
                        )
            
            self.ansatz(num_qubits = num_qubits, params = params[j, 1:], wires = wires )

    def kernel_function(self, x1, x2, num_qubits, wires, params = np.asarray([])):
        self.variational_circuit(
						x = x1,
					 	params = params[0],
						num_qubits = num_qubits,
						wires = wires
					   )
	
        adjoint_variational_circuit = qml.adjoint(self.variational_circuit)

        adjoint_variational_circuit(
                            x = x2, 
                            params = params[1],
                            num_qubits = num_qubits,
                            wires = wires
                            )
            
        return qml.probs(wires=wires)
    
    @qml.qnode(dev)
    def kernel(self, x1, x2, params):
        return self.kernel_circuit(x1=x1, x2=x2, params=params, self.wires, self.num_qubits)

    
