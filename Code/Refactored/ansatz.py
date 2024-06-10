import pennylane as qml
from pennylane import numpy as np

def efficient_su2(num_qubits, params, wires):
	for i in range(num_qubits):
		qml.RX(params[0, i, 0], wires=wires[i])
		qml.RY(params[0, i, 1], wires=wires[i])

	for i in range(num_qubits - 1):
		qml.CNOT(wires=[wires[i], wires[i + 1]])
    
	qml.CNOT(wires=[wires[num_qubits - 1], wires[0]])
    
	for i in range(num_qubits):
		qml.RX(params[0, i, 2], wires=wires[i])
		qml.RY(params[0, i, 3], wires=wires[i])

def basic(num_qubits, params, wires):

	for i in range(num_qubits):
		qml.RX(params[0, i], wires=wires[i])
		qml.RY(params[1, i], wires=wires[i])

	qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[2, 0])
