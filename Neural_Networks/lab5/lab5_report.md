# Lab 5: Transformer Neural Network for Machine Translation

**Course:** Neural Networks and Their Applications  
**Student:** Bichurina S.P., Group 09-261, 3rd year  
**University:** Kazan (Volga Region) Federal University, Institute of Computational Mathematics and Information Technologies, Department of Information Systems

---

## Task Description

Train a Transformer-based neural network using PyTorch for the task of machine translation using a parallel text corpus (English–Russian). Analyze model quality for different values of `d_model` and `heads_count` parameters.

## Mathematical Foundations

1. **Query, Key, Value computation** — The input is projected into three representations: Q = X_Q · W_Q, K = X_K · W_K, V = X_V · W_V, where W_Q, W_K, W_V are learned weight matrices.
2. **Attention weights** — Raw attention scores are computed as e_ij = Q_i · K_j^T, then normalized via softmax to produce attention weights.
3. **Context vector** — A weighted sum of the Value tensor using the attention weights.
4. **Positional encoding** — Sinusoidal positional encoding is used: sine for even indices and cosine for odd indices of the embedding vector, parameterized by position and dimension. These values are added to token embeddings to provide sequence position information.

## Training Setup

Training was performed on English–Russian parallel texts over 20 epochs. The model quality was evaluated across different configurations of `d_model` (embedding dimension) and `data_count` (number of training samples).

## Experimental Results

### d_model = 128, data_count = 1500
- Loss decreased from 5.241 (epoch 1) to 3.789 (epoch 19)
- Predictions consisted of disconnected individual words and punctuation marks

### d_model = 256, data_count = 1500
- Loss decreased from 5.817 (epoch 0) to 3.336 (epoch 19)
- The network began translating individual phrases within sentences more accurately

### d_model = 512, data_count = 1500
- Loss decreased from 5.915 (epoch 0) to 1.830 (epoch 19)
- Significant improvement — certain parts of sentences were translated correctly

### d_model = 128, data_count = 3000
- Loss decreased from 5.227 (epoch 1) to 4.105 (epoch 19)
- Increasing data helped reduce loss, but the low d_model still resulted in very poor translation quality with a limited vocabulary

### d_model = 512, data_count = 3000 (best result)
- Loss decreased from 4.659 (epoch 1) to **1.525** (epoch 19)
- Best translation quality observed. For example, the sentence *"What am I to do? Tell me, what am I to do?"* was translated as *"Но что же делать, что же делать?"*, which conveys the correct meaning. The model demonstrated an ability to translate short sentences composed of commonly used words.

## Conclusion

The experiments show that both `d_model` and `data_count` significantly affect translation quality in the Transformer model. Increasing `d_model` (the feature space dimensionality) leads to a substantial reduction in loss and improved translation accuracy. Increasing `data_count` (the number of training samples) also contributes to lower loss and better results.

The best performance was achieved with d_model = 512 and data_count = 3000, yielding the lowest loss and adequate translation of simple, frequently used sentences. However, the model still has limitations, particularly at low d_model values where translation quality is very poor and the vocabulary is restricted.

Recommendations for further improvement include: continuing to increase data_count at optimal d_model values, experimenting with other hyperparameters, applying advanced training techniques, and diversifying the training corpus.
