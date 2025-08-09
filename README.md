GPT-Style Transformer Language Model (From Scratch)

This project is an implementation of a GPT-style Transformer language model built entirely from scratch in PyTorch.
It includes:
	•	A Bigram Language Model for understanding token prediction basics.
	•	A Transformer-based GPT model with multi-head self-attention, feedforward layers, and positional embeddings.
	•	Full training loop with evaluation and text generation.

⸻

 Features
	•	Character-level tokenization for raw text input.
	•	Configurable hyperparameters for batch size, learning rate, layers, heads, and embeddings.
	•	Multi-Head Self-Attention mechanism.
	•	Dropout regularization for reducing overfitting.
	•	AdamW optimizer for stable training.
	•	Generates coherent text based on learned patterns.

⸻

📂 Project Structure

  📁 NanoGPT-From-Scratch
│── Nano_GPT.py               # Jupyter Notebook with complete implementation
│── input.txt                  # Training dataset (Tiny Shakespeare or other text)

How It Works
	1.	Data Preprocessing:
	  •	Reads raw text (input.txt).
	  •	Creates character-to-integer and integer-to-character mappings.
	  •	Splits dataset into train and validation sets.
	2.	Model Architecture:
	  •	Token and positional embeddings.
	  •	Multi-head self-attention.
	  •	Feedforward layers.
	  •	Layer normalization.
	  •	Final linear layer projecting to vocabulary size.
	3.	Training:
	  •	Uses AdamW optimizer.
	  •	Evaluates loss periodically on train/val sets.
	  •	Learns to predict the next token given a sequence.
	4.	Text Generation:
	  •	Starts with a given prompt (or empty context).
	  •	Predicts one token at a time.
	  •	Samples from probability distribution for creativity.

 Running on Apple Silicon (M1/M2) GPU

PyTorch supports the Metal Performance Shaders (MPS) backend for Apple Silicon GPUs.
	1.	Install PyTorch with MPS Support
      pip install torch torchvision torchaudio
     
  2.	Enable MPS in Code
      In the notebook, set the device:
      device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  3. Verify is MPS is available 
     import torch
     print(torch.backends.mps.is_available())  # Should return True

  4.	Run Training
      The model will automatically use the GPU when device='mps'.

Example Training Output
25.4 M parameters
step 0:     train loss 11.3376, val loss 11.3179
step 5000:  train loss 7.9628,  val loss 7.9739
step 10000: train loss 5.9162,  val loss 5.9487
step 19999: train loss 3.9739,  val loss 4.0487

