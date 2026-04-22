from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(1)
qc.h(0)
print(qc.draw())

qc.measure_all()

sim = AerSimulator()
result = sim.run(qc).result()
print(result.get_counts())