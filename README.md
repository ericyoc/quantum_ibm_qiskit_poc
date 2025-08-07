# Quantum Cryptographic Security: RSA Breaking Demo

A demonstration of quantum cryptanalysis combining **Shor's Algorithm** for RSA factorization with **Quantum Neural Networks** for cryptographic security assessment, running on real IBM quantum computers.

## Motivating Research
Yocam, E., Rizi, A., Kamepalli, M., Vaidyan, V., Wang, Y., & Comert, G. (2024). Quantum adversarial machine learning and defense strategies: Challenges and opportunities. arXiv preprint arXiv:2412.12373. https://arxiv.org/abs/2412.12373

A. Jagatha et al., "A Novel Approach to Quantum-Resistant Selective Encryption for Agricultural Sensors with Limited Resources," 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC), Las Vegas, NV, USA, 2025, pp. 00262-00271, doi: 10.1109/CCWC62904.2025.10903955. http://ieeexplore.ieee.org/abstract/document/10903955

Bhanu Prakash, Sachin Srivastava, Rupesh Kumar et al. A Numerical and Security Analysis of RSA: From Classical Encryption to Post-Quantum Strategies, 09 April 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6347614/v1]

## ACADEMIC AND RESEARCH DISCLAIMER

**This demo is intended for academic and research purposes only.** The implementation demonstrates quantum cryptanalysis concepts using toy examples that pose no threat to real-world systems. This work is designed to:

- Educate researchers and students about quantum threats to cryptography
- Provide working examples of quantum algorithms for academic study  
- Highlight the importance of post-quantum cryptography research
- Demonstrate quantum machine learning concepts

**This code should not be used for any malicious purposes or unauthorized cryptographic attacks.** Users are responsible for complying with all applicable laws and institutional policies regarding cryptographic research.

## Results

![Quantum Cryptanalysis Results](images/visualization_20250807_035945.png)

### Training Progress and Security Assessment
The visualization shows four key components of our quantum cryptanalysis:
- **Top Left**: QNN training accuracy progression on real quantum hardware
- **Top Right**: Shor's algorithm measurement results for factoring N=15
- **Bottom Left**: Final QNN performance compared to 80% accuracy target
- **Bottom Right**: Security vulnerability assessment showing 57% vulnerability probability

## Demo Overview

This demo demonstrates the quantum threat to classical cryptography through a complete end-to-end system that:

1. **Generates RSA keys** using OpenSSL and encrypts secret messages
2. **Trains a Quantum Neural Network** (QNN) to assess cryptographic vulnerabilities
3. **Executes Shor's Algorithm** to factor RSA keys on real quantum hardware
4. **Combines quantum algorithms** to provide complete cryptanalysis
5. **Persists results** to Google Drive for analysis across sessions

## Key Quantum Concepts

### Shor's Algorithm
Shor's algorithm provides exponential speedup for integer factorization, the mathematical foundation of RSA encryption security.

**Implementation**: 
- Uses quantum superposition to test multiple factors simultaneously
- Employs Quantum Fourier Transform (QFT) for period finding
- Demonstrates why RSA will be vulnerable to large-scale quantum computers

```python
def create_shors_circuit():
    qc = QuantumCircuit(4, 4)
    # Superposition for period finding
    qc.h(0)
    qc.h(1)
    # Modular exponentiation (simplified for N=15)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cp(np.pi/2, 0, 1)
    # Inverse QFT for period extraction
    qc.h(1)
    qc.cp(-np.pi/2, 0, 1)
    qc.h(0)
    qc.measure_all()
    return qc
```

### Quantum Neural Networks (QNNs)
QNNs leverage quantum entanglement and superposition for pattern recognition in cryptographic analysis.

**Implementation**:
- **Parameterized Quantum Circuits**: Trainable quantum gates (RY, RZ rotations)
- **Quantum Entanglement**: CNOT gates create correlations impossible in classical networks
- **Variational Training**: Classical optimization of quantum parameters

```python
class QuantumNeuralNetwork:
    def create_circuit(self, params, input_features):
        qc = QuantumCircuit(self.num_qubits, 1)
        
        # Input encoding via rotation gates
        for i, feature in enumerate(input_features):
            qc.ry(feature * np.pi, i)
            
        # Variational layers with entanglement
        param_idx = 0
        for layer in range(self.num_layers):
            # Create quantum entanglement
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.num_qubits - 1, 0)  # Ring connectivity
            
            # Trainable parameters
            for i in range(self.num_qubits):
                qc.ry(params[param_idx], i)
                param_idx += 1
                qc.rz(params[param_idx], i)
                param_idx += 1
        
        qc.measure(self.num_qubits - 1, 0)
        return qc
```

### Quantum Machine Learning Pipeline
Demonstrates how quantum algorithms can assess cryptographic security beyond just factoring.

**Features**:
- **Feature Engineering**: Cryptographic parameters (key length, entropy, factor count)
- **Quantum Training**: Real quantum hardware optimization
- **Binary Classification**: Vulnerable vs. Secure key assessment

## System Architecture

```
RSA Key Gen     -->  Quantum Training  -->  Cryptanalysis
(OpenSSL)           (QNN)                   Assessment
    |                   |                       |
    v                   v                       v
Message Encrypt -->  Shor's Algorithm  -->  Results Storage
(Target)            (Factoring)             (Google Drive)
```

## Getting Started

### Prerequisites

```bash
# Install required packages
pip install qiskit
pip install qiskit[visualization] 
pip install qiskit-ibm-runtime
pip install numpy matplotlib scipy
```

### IBM Quantum Setup

1. **Create IBM Quantum Account**: Sign up at [quantum-computing.ibm.com](https://quantum-computing.ibm.com/)

2. **Get API Token**: 
   - Go to Account Settings → API Token
   - Copy API token

3. **Store Token in Google Colab**:
   ```python
   from google.colab import userdata
   # Store as 'QUANTUM_API_KEY' in Colab secrets
   ```

### Running the System

```python
# Complete execution in one command
if __name__ == "__main__":
    success = main()
    if success:
        print("All systems completed successfully!")
    else:
        print("Partial completion - check error messages above")
```

## Expected Output

### Training Progress
```
==================================================
STEP 3: QUANTUM NEURAL NETWORK TRAINING
==================================================
Backend selected: ibm_brisbane
Queue: 890 jobs
QNN: 5 qubits, 20 parameters

Starting quantum training (4 iterations)...
Iter | Train Loss | Train Acc | Val Loss | Val Acc | Time(s) | Cumulative
---------------------------------------------------------------------------
   1 |      0.704 |     0.500 |    0.601 |   0.750 |    73.5 |       73.5
   2 |      0.690 |     0.500 |    0.651 |   0.583 |    68.8 |      142.3
   3 |      0.713 |     0.500 |    0.673 |   0.417 |   102.0 |      244.3

Training completed in 723.6 seconds
Final test evaluation...
Test accuracy: 0.750 (75.0%)
Total quantum time: 758.8 seconds
80% target: CLOSE
```

### Shor's Algorithm Results
```
==================================================
STEP 4: SHOR'S ALGORITHM EXECUTION
==================================================
Factoring target: N = 15
Shor's circuit: 4 qubits, depth 7
Executing Shor's algorithm...
Most frequent result: 1110
Factorization successful: 15 = 3 x 5
```

### Security Assessment
```
==================================================
STEP 5: QNN SECURITY ASSESSMENT
==================================================
Analyzing N=15 security features: [0.333, 0.5, 0.2, 0, 0.3]
Vulnerability probability: 0.570
QNN Classification: VULNERABLE
Confidence: 0.570
```

## Performance Analysis

### Quantum Resource Usage
- **Quantum Time**: ~12.6 minutes actual execution
- **Queue Wait**: Variable (890 jobs in example run)
- **QNN Accuracy**: 75% (close to 80% target)
- **Shor's Success**: 100% factorization success
- **Budget Efficiency**: 252.9% of 5-minute allocation

### Key Metrics
| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| QNN Accuracy | 80% | 75% | Close |
| Shor's Factorization | Success | Success | Complete |
| RSA Key Breaking | Demo | Complete | Complete |
| End-to-End Integration | Full | Complete | Complete |

## Code Structure

### Main Components

```python
# 1. Environment Setup
DRIVE_PATH, service = setup_environment()

# 2. Cryptographic Setup  
crypto_data = create_rsa_encryption_demo()

# 3. Dataset Creation
train_data, val_data, test_data = create_training_dataset()

# 4. QNN Training
qnn_results, qnn = train_qnn_for_80_percent_accuracy(service, train_data, val_data, test_data)

# 5. Shor's Algorithm
shor_counts, target_n = execute_shors_algorithm(service, qnn)

# 6. Security Assessment
vulnerability_prob, classification = perform_qnn_security_assessment(qnn, qnn_results)

# 7. Results Storage
save_results_to_drive(DRIVE_PATH, crypto_data, qnn_results, shor_counts, security_assessment)
```

### Dataset Features
The cryptographic dataset uses 5 key features for security assessment:
- **Key Length** (normalized): Bit length of cryptographic key
- **Factor Count** (normalized): Number of prime factors
- **Entropy Score**: Randomness quality of key generation
- **Known Weaknesses**: Presence of implementation flaws
- **Implementation Quality**: Overall security of key usage

## Scientific Impact

### Quantum Advantage Demonstration
1. **Factorization Speedup**: Shor's algorithm achieves exponential speedup over classical factoring
2. **Quantum ML Benefits**: QNN leverages quantum entanglement for pattern recognition
3. **Real Hardware Validation**: Results from actual IBM quantum processors, not simulation

### Cryptographic Implications
- **Current Threat**: Small RSA keys already vulnerable
- **Future Threat**: Larger keys will be vulnerable as quantum computers scale
- **Mitigation Need**: Urgent transition to post-quantum cryptography required

## Limitations and Future Work

### Current Limitations
- **Small Key Size**: Demo limited to N=15 due to quantum hardware constraints
- **NISQ Era**: Current quantum computers are noisy and limited
- **Queue Dependencies**: Execution time varies with IBM Quantum usage

### Future Enhancements
- **Larger Keys**: Target 512-bit, 1024-bit RSA keys as hardware improves
- **Error Correction**: Implement quantum error correction for reliability
- **Advanced QNNs**: Deeper networks with more sophisticated architectures
- **Hybrid Algorithms**: Classical-quantum hybrid optimization

## File Structure

```
quantum_files/
├── complete_cryptanalysis_YYYYMMDD_HHMMSS.json    # Full results
├── latest_cryptanalysis.json                      # Most recent results
├── dataset_YYYYMMDD_HHMMSS.json                  # Training data
├── visualization_YYYYMMDD_HHMMSS.png             # Result plots
└── README.md                                      # This file
```

## Research Applications

### Academic Use Cases
- **Quantum Algorithm Education**: Complete working examples of Shor's algorithm
- **Quantum ML Research**: Real quantum neural network implementation
- **Cryptanalysis Studies**: End-to-end security assessment methodology

### Industry Applications
- **Security Assessment**: Evaluate cryptographic system vulnerabilities
- **Post-Quantum Migration**: Understand quantum threat timeline
- **Quantum Readiness**: Prepare for quantum computing era

## Ethical Considerations

### Responsible Use Policy
This research tool is provided for legitimate academic and research purposes only. Users must:

- Use only for educational, research, or authorized security assessment purposes
- Comply with all applicable laws and regulations
- Respect intellectual property and privacy rights
- Not use for unauthorized access or malicious attacks
- Follow institutional ethics guidelines for cryptographic research

### Security Implications
- **Current Impact**: Minimal - targets only very small keys for demonstration
- **Future Impact**: Significant - will affect all RSA-based systems as quantum computers scale
- **Recommendation**: Begin post-quantum cryptography transition planning

## References and Further Reading

1. **Shor, P.W.** (1994). "Algorithms for quantum computation: discrete logarithms and factoring"
2. **IBM Quantum Documentation**: [qiskit.org](https://qiskit.org/)
3. **NIST Post-Quantum Cryptography**: [csrc.nist.gov/projects/post-quantum-cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
4. **Quantum Machine Learning**: Schuld & Petruccione (2018)

## Support and Contributions

### Getting Help
- **IBM Quantum Community**: [community.qiskit.org](https://community.qiskit.org/)
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Join quantum computing discussions and share results

## Acknowledgments

- **IBM Quantum Team** for providing access to quantum hardware
- **Qiskit Community** for quantum computing tools and support
- **Google Colab** for cloud computing infrastructure
- **OpenSSL Project** for cryptographic implementations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



---

**Ready to explore quantum cryptanalysis for research? Clone this repository and run on IBM Quantum hardware for educational purposes.**
