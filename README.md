# Sci-Fi Story Generator with SentencePiece

This project implements a Transformer-based language model for generating sci-fi stories. It uses SentencePiece for subword tokenization and is trained on text from Project Gutenberg's "The Time Machine" by H.G. Wells.

## Features

* **Transformer Architecture:** Utilizes a Transformer model for sequence-to-sequence learning.
* **SentencePiece Tokenization:** Employs SentencePiece for efficient and language-agnostic subword tokenization.
* **Sci-Fi Text Generation:** Generates coherent sci-fi stories based on learned patterns.
* **Training and Inference Scripts:** Includes scripts for training the model and generating stories.
* **Device Agnostic:** Supports training and inference on CPU, CUDA, and MPS.

## Getting Started

### Prerequisites

* Python 3.8 or higher
* PyTorch
* SentencePiece
* Requests
* Tqdm

Install the required packages:

```bash
pip install -r requirement.text
```

### Training the Model

Clone the Repository:

```bash
git clone https://github.com/macroadster/story-teller.git
cd story-teller
```

Run the Training Script:

```bash
python train.py
```

This script will:

  Download the sci-fi text from Project Gutenberg.
  Train a SentencePiece tokenizer.
  Generate training data.
  Train the Transformer model.
  Save the trained model (transformer_llm.pth) and SentencePiece model (sentencepiece_model.model).
  Generate sample story. (Inference)

Run the Inference Script:

```bash
python inference.py
```

This script will:

  Load the trained model and SentencePiece model.
  Generate a story.
  You can modify the inference.py script to change the starting text, maximum length, and temperature of the generated stories.

### Customization

* Training Data: To train the model on different text, modify the fetch_sci_fi_text function in train.py to fetch your desired text data.
* Model Parameters: Adjust the model's hyperparameters (e.g., d_model, num_heads, num_layers) in train.py to experiment with different model configurations.
* Vocabulary Size: Modify the vocab_size parameter in train_sentencepiece_tokenizer to change the size of the SentencePiece vocabulary.
* Generation Parameters: Change the temperature and max_length parameters in the generate_sci_fi_story function to control the generation process.
* Device selection: Change the device string, to use cuda, mps or cpu.

#### Model Files

* transformer_llm.pth: Contains the trained model's state dictionary.
* sentencepiece_model.model: Contains the trained SentencePiece model.

### Code Structure

* train.py: Training script.
* inference.py: Inference script.

* MultiHeadAttention, PositionWiseFeedForward, TransformerBlock, TransformerLLM: Model architecture components.
* fetch_sci_fi_text, train_sentencepiece_tokenizer, tokenize, detokenize, generate_training_data, TextDataset, custom_collate: Data processing and utility functions.
* generate_sci_fi_story: Story generation function.

### Future Improvements

* Implement early stopping to prevent overfitting.
* Add more comprehensive evaluation metrics.
* Experiment with different model architectures.
* Fine-tune the model on a larger and more diverse dataset.
* Add better handling of sentence boundaries and punctuation.
* Add command line arguments to the scripts.
* Add a configuration file.
