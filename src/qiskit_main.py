import json
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def main():
    with open("data/pca/training_pca.json") as f:
        pca_data = json.load(f)
    lm_files = list(Path("data/landmarks/").glob("*.json"))
    