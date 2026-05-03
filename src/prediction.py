import json
from pathlib import Path
import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector



def main():
    #saveds params
    if Path("models/quant/trained_params.json").exists():
        with open("models/quant/trained_params.json") as f:
            params = json.load(f)
    else:
        raise Exception("File not found: models/quant/trained_params.json.")

    #saved pca data
    if Path("data/pca/test_pca.json").exists():
        with open("data/pca/test_pca.json") as f:
            pca_data = json.load(f)
    else: 
        raise Exception("File not found: data/test_pca.json")

    #-----------------------------------
    #INPUT ENCODING --------------------------------------------------
    #Varitational Layer --------------------------------------------
    test_pca = [entry["PCA_Data"] for entry in pca_data]
    min_val = params["min_val"]
    max_val = params["max_val"]
    enc_params = [Parameter(f"enc_{i}") for i in range(10)]   #Encoding Placeholder 
    var_params = [Parameter(f"theta_{i}") for i in range(10)] #Variational Placeholder
    qc = QuantumCircuit(10)

    for i in range(10): #encoding PCA data -> rotations
        qc.ry(enc_params[i], i) #qc.ry(angle, quibit)  
        qc.ry(var_params[i], i)
    
    all_params = params["params"]
    quantum_params = all_params[:10] #first 10
    weights = np.array(all_params[10:640]).reshape(10,63) #next 630 -> 10x63 grid
    bias = np.array(all_params[640:])    # last 63 
    result = []
    for idx_frame in range(len(test_pca)):
        if idx_frame % 100 == 0:
            print(f"Processing frame {idx_frame}/{len(test_pca)}")
        param_values = {}
        for i in range(10):
            param_values[enc_params[i]] = ((test_pca[idx_frame][i] - min_val) / (max_val - min_val) * math.pi)
            param_values[var_params[i]] = quantum_params[i] #same for all samples
        bound_qc = qc.assign_parameters(param_values)
        # actual = np.array(test_pca[idx_frame]).flatten() #63 nums
        #Expectation Vals -----------------------------------------------
        sv = Statevector(bound_qc)
        expectations = []
        for i in range(10):
            z_label = ['I'] * 10 #do nothing on all 10x quibits
            z_label[i] = 'Z' #measure quibit i
            op = SparsePauliOp(''.join(z_label))
            exp_val = sv.expectation_value(op).real
            expectations.append(exp_val)
        predictions = np.dot(expectations, weights) + bias
        landmarks = predictions.reshape(21, 3).tolist() #21 joints, each[x,y,z]
        result.append({
            "landmarks": landmarks,
            "timestamp_ms": pca_data[idx_frame]["timestamp_ms"],
            "videopath": pca_data[idx_frame]["video_path"]
        })

    with open("data/quant-gen-vids/test.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()