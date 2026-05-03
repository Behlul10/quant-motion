import json
from pathlib import Path
import numpy as np
import math
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector


def qiskit_main():
    #DATA LOADING & Matching ------------------------------
    with open("data/pca/training_pca.json") as f:
        pca_data = json.load(f)
    lm_files = list(Path("data/landmarks/").glob("*.json"))
    X_train = [] #PCA Features
    Y_train = [] #Landmakrs
    lm_lookup = {}
    N_QUIBITS = 10
    N_LANDMARKS = 63 # 21 joints x 3 coords
    BATCH_SIZE = 200

    #looping through landmarks and getting a dictionary
    for file in lm_files:
        video_name = file.stem
        with open(file) as f:
            entries = json.load(f)
        for entry in entries:
            key = (video_name, entry["timestamp"])
            lm_lookup[key] = entry["landmarks"]
    
    #which PCA data matches with the timestamp?
    for pca_entry in pca_data:
        video_name = Path(pca_entry["video_path"]).stem
        timestamp = pca_entry["timestamp_ms"]
        key = (video_name, timestamp)
        if key in lm_lookup:
            X_train.append(pca_entry["PCA_Data"]) #10 nums
            Y_train.append(lm_lookup[key]) #landmarks
    print(len(X_train))
    #--------------------Circuit Building ----------------------------
    #INPUT ENCODING --------------------------------------------------
    #Varitational Layer --------------------------------------------
    #random params to adjust
    X_array = np.array(X_train)
    min_val = X_array.min()
    max_val = X_array.max()
    enc_params = [Parameter(f"enc_{i}") for i in range(N_QUIBITS)]   #Encoding Placeholder 
    var_params = [Parameter(f"theta_{i}") for i in range(N_QUIBITS)] #Variational Placeholder
    qc = QuantumCircuit(N_QUIBITS)

    for i in range(N_QUIBITS): #encoding PCA data -> rotations
        qc.ry(enc_params[i], i) #qc.ry(angle, quibit)  
        qc.ry(var_params[i], i)
    
    if Path("models/quant/trained_params.json").exists():
        with open("models/quant/trained_params.json") as f:
            saved = json.load(f)
        random_nums = np.array(saved["params"]) #resume from saved
    else:
        #starting Values with seed 
        np.random.seed(42)
        random_nums = np.random.randn(N_QUIBITS + N_QUIBITS * N_LANDMARKS + N_LANDMARKS)
        

    def compute_loss(all_params):
        quantum_params = all_params[:N_QUIBITS] #first N Quibits
        weights = all_params[N_QUIBITS:N_QUIBITS+N_QUIBITS*N_LANDMARKS].reshape(N_QUIBITS,N_LANDMARKS) #next 630 -> N_QuibitsxN_landmarks grid
        bias = all_params[N_QUIBITS + N_QUIBITS * N_LANDMARKS:] 
        
        #Varitational Layer --------------------------------------------
        # param_values = {var_params[i]: quantum_params[i] for i in range(10)}
        total_loss = 0
        
        for sample_idx in range(BATCH_SIZE):
            param_values = {}
            for i in range(N_QUIBITS):
                param_values[enc_params[i]] = ((X_train[sample_idx][i] - min_val) / (max_val - min_val) * math.pi)
                param_values[var_params[i]] = quantum_params[i] #same for all samples
            bound_qc = qc.assign_parameters(param_values)
    
            actual = np.array(Y_train[sample_idx]).flatten() #63 nums
            #Expectation Vals -----------------------------------------------
            sv = Statevector(bound_qc)
            expectations = []
            for i in range(N_QUIBITS):
                z_label = ['I'] * N_QUIBITS #do nothing on all 10x quibits
                z_label[i] = 'Z' #measure quibit i
                op = SparsePauliOp(''.join(z_label))
                exp_val = sv.expectation_value(op).real
                expectations.append(exp_val)
            
            #Prediction & Loss
            predictions = np.dot(expectations, weights) + bias
            nums = (predictions - actual ) ** 2
            total_loss += (sum(nums) / len(nums)) #loss is avg
            
        return total_loss / BATCH_SIZE
    result = minimize(compute_loss, random_nums, method='COBYLA', options={'maxiter':200})
    print("LOSSES: ", result.fun)
    
    with open("models/quant/trained_params.json", "w") as f:
        json.dump({
            "params": result.x.tolist(),
             "loss": result.fun,
             "min_val": float(min_val),
             "max_val": float(max_val)
             }, f)

if __name__ == "__main__":
    qiskit_main()