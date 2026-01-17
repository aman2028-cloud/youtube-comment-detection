#  YouTube Toxic Comment Detection using DistilBERT

Toxic and abusive comments are a major challenge on online platforms such as YouTube, negatively impacting user experience and community safety. Automatically detecting such content is a key Natural Language Processing (NLP) task with applications in **content moderation, digital safety, and social media analytics**. The problem is challenging due to informal language, sarcasm, overlapping toxicity categories, and highly imbalanced data.

In this project, I implemented a **Transformer-based multi-label text classification system** for detecting toxic behavior in YouTube comments. The model is fine-tuned using **DistilBERT**, a lightweight yet powerful transformer architecture, and deployed as an **interactive Streamlit web application** for real-time inference.

The project covers the **complete ML lifecycle**: dataset exploration, preprocessing, model fine-tuning, evaluation, and deployment.

---
### Toxicity Labels

The model predicts probabilities for the following six categories:

- Toxic  
- Severe Toxic  
- Obscene  
- Threat  
- Insult  
- Identity Hate  

Each label is modeled independently using sigmoid activations rather than softmax.

---

## Dataset

- **Source:** Kaggle (Jigsaw Toxic Comment Classification Challenge)  
- **Size:** ~150,000 comments  
- **Language:** English  
- **Label Type:** Multi-label (binary per class)  
- **Key Challenges:**  
  - Severe class imbalance  
  - Noisy, user-generated text  
  - Overlapping labels (e.g., toxic + obscene)  

---

## Data Preprocessing

Before training, the text data undergoes several preprocessing steps:

- Removal of URLs, mentions, hashtags, and special characters  
- Lowercasing and whitespace normalization  
- Tokenization using **DistilBERT Tokenizer**  
- Padding and truncation to a fixed maximum sequence length (128 tokens)  

Transformer-based tokenization allows the model to capture contextual meaning without aggressive text normalization.

---

## Model Architecture: DistilBERT for Multi-label Classification

DistilBERT is a **compressed version of BERT** that retains ~97% of BERT’s language understanding while being **40% smaller and faster**.

### Architecture Details

- **Base Model:** `distilbert-base-uncased`  
- **Embedding Dimension:** 768  
- **Transformer Layers:** 6  
- **Classification Head:**  
  - Linear layer → 6 output neurons  
  - Sigmoid activation for multi-label probabilities  
- **Loss Function:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss)  

This setup allows the model to assign independent probabilities to each toxicity category.

---

## Training

- **Framework:** PyTorch + HuggingFace Transformers  
- **Optimizer:** AdamW  
- **Learning Rate:** 2e-5  
- **Batch Size:** 16  
- **Epochs:** 2  
- **Regularization:** Dropout in classifier head  
- **Hardware:** Google Colab (GPU)  

The model was fine-tuned end-to-end, including the classification head, to adapt the pretrained language representations to toxic language patterns.

---

### Observations

- High recall across most toxicity categories  
- Strong performance on frequent labels such as *toxic* and *obscene*  
- Reasonable detection of rare labels like *threat* and *identity hate*  
- Multi-label predictions align well with real-world usage

---

The trained model is deployed using **Streamlit**, enabling real-time toxicity analysis.

## Tech Stack

- **Language:** Python  
- **Deep Learning:** PyTorch  
- **NLP:** HuggingFace Transformers  
- **Model:** DistilBERT  
- **Web Framework:** Streamlit  
- **Evaluation:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  

---

