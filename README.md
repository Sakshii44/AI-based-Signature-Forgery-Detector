 AI-Based Signature Forgery Detector using Siamese Network

This project is a deep learning-based solution for signature forgery detection using a Siamese Neural Network. It takes two signature images as input and determines whether they belong to the same person (genuine) or different people (forged).

---

Features

- Signature verification using a trained Siamese Network  
- Uses contrastive loss to learn similarity between image pairs  
- High accuracy and real-time performance  
- Custom dataset support with genuine and forged signatures  
- Streamlit-based web UI for real-time signature comparison (optional)  

---

Project Structure

├── dataset/
│ ├── genuine/
│ └── forged/
├── siamese_model.py # Model architecture
├── siamese_train.py # Training script
├── siamese_eval.py # Evaluation script
├── siamese_utils.py # Helper functions
├── streamlit_app.py # Optional: Streamlit interface
├── requirements.txt
└── README.md

---

Dataset

You can use:
- Your own dataset structured as:
dataset/
├── person1/
├── genuine/
├── forged/
├── person2/
...

- Or use benchmark datasets like:
- [CEDAR Signature Dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar)
- [GPDS Synthetic Signature Dataset](https://www.gpds.ulpgc.es/)
