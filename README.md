# Modular_Text_Summarisation
Rouge_score comparison for baseline| with Preprocessing| Fine-tuning+Preprocessing using Pegasus-xsum on Samsum dataset.
This repository demonstrates how to perform extractive summarization using the **Pegasus** model, specifically the **Pegasus-XSum** checkpoint, on the **SAMSum** dataset. The project includes the baseline performance evaluation using the **ROUGE** scores, data preprocessing, and model evaluation. Fine-tuning steps will be implemented in future versions of this project.
## Project Structure

```
├── main.py                    # Main script to load and evaluate the model
├── requirements.txt            # List of required Python packages
├── config.yaml                 # Configuration file for project settings
├── summarisation/              # Folder containing the core summarization logic
│   ├── pipeline.py             # Core pipeline for the summarization process
│   ├── data_preprocessing.py   # Data preprocessing scripts
├── visualisation.ipynb         # Notebook for visualizing results
└── README.md                   # Project documentation (this file)

```



---

### **10. Future Improvements**



## Future Improvements

- **Fine-tuning**: Implement LoRA-based fine-tuning for improved task-specific performance.
- **Optimization**: Reduce memory usage during training by offloading parts of the model to CPU when not in use.
- **Evaluation**: Use additional evaluation metrics such as **BLEU** or **METEOR** for further model comparison.

## ROUGE Score Comparison

The following table compares the ROUGE scores for the baseline model and the model after preprocessing. These scores provide an indication of the model's performance on the SAMSum dataset.

| **Model**              | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|------------------------|-------------|-------------|-------------|
| **Baseline**            | 0.1708      | 0.0137      | 0.1292      |
| **After Preprocessing/Postprocessing(minimal)** | 0.1858      | 0.0042      | 0.1373      |

- **ROUGE-1**: Measures the overlap of unigrams between the model's output and the reference summary.
- **ROUGE-2**: Measures the overlap of bigrams between the model's output and the reference summary.
- **ROUGE-L**: Measures the longest common subsequence (LCS) between the model's output and the reference summary.
**Aggressive preprocessing such as speaker_id_removal, salutaions removal, affect the context understanding of the pegasus-xsum model more on samsum dataset which is a Conversational Dataset.**

  
