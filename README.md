# GraphSAGE on PubMed (PyTorch Geometric)

This repository contains an end-to-end implementation of **GraphSAGE** on the **PubMed** citation dataset using **PyTorch Geometric**.  
The project evaluates three different GraphSAGE aggregators:

- **Mean Aggregator**
- **Pooling (Max) Aggregator**
- **LSTM Aggregator** (not fully executed due to GPU memory constraints)

The model uses **mini-batch training with neighbor sampling** for scalability and includes:

- Training loops for each aggregator  
- Validation and test evaluation (Accuracy & Macro F1)  
- t-SNE visualization of learned embeddings  
- Subgraph visualization of PubMed  
- Saved outputs for additional analysis  

---

## 🔍 Results

### **Mean Aggregator**
- **Best Validation Accuracy:** 0.7940  
- **Best Test Accuracy:** 0.7760  
- **Final Test Accuracy:** 0.7720  
- **Final Macro F1:** 0.7650  

### **Pooling (Max) Aggregator**
- **Best Validation Accuracy:** 0.7980  
- **Best Test Accuracy:** 0.7720  
- **Final Test Accuracy:** 0.7790  
- **Final Macro F1:** 0.7765  

### **Why LSTM Aggregator Was Not Used**
The **LSTM aggregator** requires significantly more GPU memory than Mean or Pooling aggregators because:

1. It processes each neighbor *as a sequence*, with hidden states stored for every neighbor during backpropagation.
2. PubMed contains ~20k nodes and dense neighborhoods, so even with neighbor sampling, memory grows rapidly.
3. On a **6GB GPU**, the combination of:
   - neighbor sampling  
   - two GraphSAGE layers  
   - LSTM hidden states  
   - large batch sizes  
   - 128-dim embeddings  

   caused repeated **CUDA Out-of-Memory errors (~6–8 GB required)**.

For this reason, only the Mean and Pooling aggregators were successfully trained.



