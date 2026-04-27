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
    enc_params = [Parameter(f"enc_{i}") for i in range(10)]   #Encoding Placeholder 
    var_params = [Parameter(f"theta_{i}") for i in range(10)] #Variational Placeholder
    qc = QuantumCircuit(10)

    for i in range(10): #encoding PCA data -> rotations
        # scaled_pca_val = (X_train[0][i] - min_val) / (max_val - min_val) * math.pi
        qc.ry(enc_params[i], i) #qc.ry(angle, quibit)  
        qc.ry(var_params[i], i)
    
    #starting Values with seed 
    np.random.seed(42)
    random_nums = np.random.randn(703)
    
    def compute_loss(all_params):
        quantum_params = all_params[:10] #first 10
        weights = all_params[10:640].reshape(10,63) #next 630 -> 10x63 grid
        bias = all_params[640:]    # last 63 
        
        #Varitational Layer --------------------------------------------
        # param_values = {var_params[i]: quantum_params[i] for i in range(10)}
        total_loss = 0
        
        for sample_idx in range(10):
            param_values = {}
            for i in range(10):
                param_values[enc_params[i]] = ((X_train[sample_idx][i] - min_val) / (max_val - min_val) * math.pi)
                param_values[var_params[i]] = quantum_params[i] #same for all samples
            bound_qc = qc.assign_parameters(param_values)
    
            actual = np.array(Y_train[sample_idx]).flatten() #63 nums
            #Expectation Vals -----------------------------------------------
            sv = Statevector(bound_qc)
            expectations = []
            for i in range(10):
                z_label = ['I'] * 10 #do nothing on all 10x quibits
                z_label[i] = 'Z' #measure quibit i
                op = SparsePauliOp(''.join(z_label))
                exp_val = sv.expectation_value(op).real
                expectations.append(exp_val)
            
            #Prediction & Loss
            predictions = np.dot(expectations, weights) + bias
            nums = (predictions - actual ) ** 2
            total_loss += (sum(nums) / len(nums)) #loss is avg
            
        return total_loss / 10
    result = minimize(compute_loss, random_nums, method='COBYLA', options={'maxiter':200})
    print("LOSSES: ",result.fun)

if __name__ == "__main__":
    qiskit_main()