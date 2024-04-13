from pennylane import numpy as np
import pennylane as qml
from sklearn.svm import SVC


dev = qml.device("default.qubit", wires=1, shots=None)
wires = dev.wires.tolist()

def initqml(qubitType = "default.qubit", 
			wires = 1,
			shots = None
		   ):
	
	dev = qml.device(qubitType, wires=wires, shots=shots)
	wires = dev.wires.tolist()

def trainingLayers(x, 
				   params, 
				   wires, 
				   i0=0, 
				   inc=1
				  ):
	i = i0
	for j, wire in enumerate(wires):
		qml.Hadamard(wires=[wire])
		qml.RZ(x[i % len(x)], wires=[wire])
		qml.RZ(x[i], wires=[wire])
		i += inc
		qml.RY(params[0, j], wires=[wire])
	qml.broadcast(unitary=qml.CRZ, 
			   	  pattern="ring", 
				  wires=wires, 
				  parameters=params[1]
				 )
	
def variationalCircuit(x, 
					   params, 
					   wires
					  ):
	
	for j, layer_params in enumerate(params):
		trainingLayers(x, layer_params, wires, i0=j * len(wires))

def ConjTransposeVirationalCircuit():
	return qml.adjoint(variationalCircuit)

def qkernelCircuitSubsampling(x1, 
							  x2, 
							  params
							 ):
	variationalCircuit(x1, params, wires)
	ConjTransposeVirationalCircuit()
	return qml.probs(wires=wires)

def qkernel(x1, 
			x2, 
			params, 
			kernel = "subsampling"
		   ):
	
	return qkernelCircuitSubsampling(x1, x2, params)[0]

def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2 * np.pi, 
							 (num_layers, 2, num_wires), requires_grad=True
							)




