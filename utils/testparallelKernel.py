import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3, shots=None)
wires = dev.wires.tolist()

@qml.qnode(dev)
def kernel():
    
    qml.Hadamard(wires=0)
    qml.RZ(1.45678978978977778888, wires=0)
    qml.RZ(1.45678978978977778888, wires=0)
    qml.Hadamard(wires=0)

    
    qml.CNOT([0, 2])
    qml.PauliX(2)
    
    qml.SWAP([0, 1])
    qml.PauliX(1)
    
    a = qml.probs(0)
    b = qml.probs(1)
    c = qml.probs(2)



    return [a, b, c]

print("----------------------------------------------------------------------------------------------------")
drawer = qml.draw(kernel)
print(drawer())
print("----------------------------------------------------------------------------------------------------")
print('Distance between 1st and 2nd Data Points', kernel())
print("----------------------------------------------------------------------------------------------------")