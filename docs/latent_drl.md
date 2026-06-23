# SYSTEM SPECIFICATION: Latent Deep Reinforcement Learning Agent for Imperfect Information Card Game

## 1. Architectural Overview
You are tasked with implementing an end-to-end Latent Reinforcement Learning Agent based on the principles of MuZero and AlphaZero, customized for a complex, imperfect-information trading card game (PTCG). 
The architecture bypasses traditional belief state determinization by conducting its tree search entirely within a continuous latent embedding space R^d. The topology of the search tree is explicitly tracked using **Action-History Hash Keys** (tuples of integers representing imaginary choice paths).

### Pipeline Flow:
Real-World History (I_0:t) --> Transformer Encoder (f_theta) --> Latent Root (h_root)
                                                                 │
                                                       ┌─────────┴─────────┐
                                                       ▼                   ▼
                                                Policy Head (pi_phi)  Value Head (V_psi)

---

## 2. Core Modules to Implement

### Module A: `networks.py` (Neural Engine)
Implement a unified PyTorch `nn.Module` managing four interconnected layers:

1. **Representation Network (Transformer Encoder) f_theta(I_0:t) -> h_t**
   * **Input:** Variable-length historical sequence of game environment states I_0:t mapped into integer entity tokens.
   * **Processing:** Pass token arrays through a Causal Transformer Encoder (`nn.TransformerEncoder`). Use global mean-pooling over the sequence dimensions followed by a linear projection layer.
   * **Output:** Continuous latent belief vector h in R^d.
2. **Dynamics Network g_omega(h_k, a_k) -> h_k+1**
   * **Input:** Current latent state h_k and a discrete action index a_k.
   * **Processing:** Concatenate h_k with an action embedding vector (`nn.Embedding`), then pass the combined vector through a shallow Multi-Layer Perceptron (MLP) with ReLU activations.
   * **Output:** Predicted subsequent imaginary latent state h_k+1 in R^d.
3. **Policy Head pi_phi(h_k) -> p_k**
   * **Input:** Latent vector h_k.
   * **Output:** Raw logits over the universal action space dimension.
4. **Value Head V_psi(h_k) -> v_k**
   * **Input:** Latent vector h_k.
   * **Output:** A single scalar v_k in [-1, +1] utilizing a `Tanh` output activation, representing win probability.

### Module B: `mcts.py` (The Search Manager)
Implement a search engine class that plans real-world choices using an action-history lookup paradigm.

* **State Space Key Definition:** Every node in the search tree is explicitly keyed by an immutable python tuple: `action_history = (a_0, a_1, ..., a_{k-1})`. The root node is the empty tuple `()`.
* **Hash Table Storage Struct:** ```python
  self.tree = {} # Key: tuple | Value: dict("A": list, "N": Tensor, "Q": Tensor, "P": Tensor)