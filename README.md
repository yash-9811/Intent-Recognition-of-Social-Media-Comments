# Intent Classification using BERT (Custom Dataset)

This project fine-tunes a pre-trained BERT model (`bert-large-uncased`) for **intent classification** based on a self-curated dataset of user comments. The model is trained to identify the intent behind a comment and provide a confidence score for its prediction.

---

## 📁 Dataset

The dataset is stored in an Excel file:  
## 📷 Dataset Screenshots

### Sample View 1
![Dataset Screenshot 1](https://github.com/yash-9811/Intent-Recognition-of-Social-Media-Comments/blob/main/dataset_ss1.png?raw=true)

### Sample View 2
![Dataset Screenshot 2](https://github.com/yash-9811/Intent-Recognition-of-Social-Media-Comments/blob/main/dataset_ss2.png?raw=true)

It contains two columns:
- `Comment` – User-generated text input.
- `Intent` – Corresponding label denoting the intent of the comment.

> Emojis (`😊`) and ellipses (`...`) are filtered out during preprocessing.

---

## 🔧 Setup & Dependencies

Ensure the following packages are installed:

```bash
pip install torch transformers scikit-learn pandas openpyxl
```

---

## 🧠 Model Architecture

- **Base Model**: `bert-large-uncased` from HuggingFace Transformers
- **Fine-Tuning**:
  - Classification head on top of BERT
  - Loss function: CrossEntropyLoss
  - Optimizer: AdamW
  - Scheduler: StepLR (decays learning rate after every epoch)
  - Epochs: 5
  - Batch Size: 32
- **Tokenizer**: `BertTokenizer` from HuggingFace

---

## 🚀 Training Pipeline

1. **Preprocessing**
   - Filter out noisy inputs (e.g., emojis).
   - Label encoding for intent classes.
   - Tokenization using BERT tokenizer (`max_length=128`, padding & truncation applied).

2. **DataLoader Creation**
   - Training and test sets created using `train_test_split`.

3. **Training Loop**
   - Training for 5 epochs with average training loss displayed per epoch.

4. **Evaluation**
   - Accuracy and classification report generated on test set.

---

## 📊 Evaluation
![Results](https://github.com/yash-9811/projects-assets/blob/main/evaluation_results.png?raw=true)

---

## 🔮 Inference

An interactive loop allows users to input comments and receive:
- **Predicted Intent**
- **Confidence Score** (based on softmax probabilities)

```python
Enter a comment (or 'exit' to quit): How do I reset my password?
Intent: Account_Help, Confidence: 0.9421
```

---

## 🛠 Customization

- 🔍 Change BERT variant (`bert-base-uncased`, `distilbert-base-uncased`, etc.)
- 📁 Replace the Excel dataset path as needed:
  ```python
  df = pd.read_excel(r'C:\Users\yashs\Desktop\Major_Data_Set_Sample.xlsx')
  ```
- 🧪 Adjust `max_seq_length`, `batch_size`, or optimizer settings as required.

---

## 📌 Notes

- The model ignores inputs containing emojis or ellipses during training and inference.
- The label encoding ensures model outputs map directly back to original intent names.

---

## 🧑‍💻 Author

Yash Srivastava

---

## 📜 License

This project is for educational/research purposes. For commercial usage, ensure compliance with BERT and dataset licensing terms.

---

