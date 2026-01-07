# Day 21 - Practical NLP with Hugging Face
 
 **Topics Covered:** Transformers Library, Pipelines, Tokenizers, Datasets, Pre-trained Models, Fine-Tuning Steps
 
 ---
 
 ## Question 1: What is Hugging Face?
 
 **Topic:** Ecosystem
 **Difficulty:** Basic
 
 ### Question
 What is Hugging Face, and why is it considered the "GitHub of NLP"?
 
 ### Answer
 
 **Hugging Face** is a platform and community that provides:
 1. **Model Hub:** Hosting 500,000+ pre-trained models (BERT, GPT, Llama, Whisper).
 2. **Datasets:** A library to access thousands of datasets easily.
 3. **Transformers Library:** An open-source Python library that unifies PyTorch and TensorFlow interfaces for state-of-the-art models.
 
 ---
 
 ## Question 2: The `pipeline()` function
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 What is the `pipeline()` function in the Transformers library? Write code to create a sentiment analysis pipeline.
 
 ### Answer
 
 **Concept:** A high-level abstraction that handles the complexity of basic NLP tasks (Raw Text -> Tokenization -> Model -> Post-processing -> Output).
 
 **Code:**
 ```python
 from transformers import pipeline
 
 # 1. Create pipeline
 classifier = pipeline("sentiment-analysis")
 
 # 2. Run on text
 result = classifier("I love using Hugging Face!")
 
 print(result)
 # [{'label': 'POSITIVE', 'score': 0.999}]
 ```
 
 ---
 
 ## Question 3: AutoTokenizer & AutoModel
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Why do we use `AutoTokenizer` and `AutoModel` instead of specific classes like `BertTokenizer`?
 
 ### Answer
 
 - **Flexibility:** They automatically detect the correct architecture from the model checkpoint string.
 - If you change your model from `"bert-base-uncased"` to `"roberta-base"`, you don't need to rewrite your code imports.
 - **Example:**
    ```python
    from transformers import AutoTokenizer
    # Automatically loads DistilBertTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ```
 
 ---
 
 ## Question 4: Tokenizing for Models
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 When tokenizing a batch of sentences for BERT, what three specific arguments are crucial?
 
 ### Answer
 
 ```python
 encoding = tokenizer(
     ["Sentence 1", "Short"],
     padding=True,      # Pad short, truncate long
     truncation=True,
     return_tensors="pt" # Return PyTorch tensors (or "tf" for TensorFlow)
 )
 ```
 
 1. **padding:** Ensures all vectors in the batch have the same length (fills with 0s).
 2. **truncation:** Cuts off text longer than the model's limit (e.g., 512).
 3. **return_tensors:** Returns the correct tensor format instead of Python lists.
 
 ---
 
 ## Question 5: Datasets Library
 
 **Topic:** Data Handling
 **Difficulty:** Basic
 
 ### Question
 How does the `datasets` library efficiently handle 100GB+ datasets (like Common Crawl) on a small laptop?
 
 ### Answer
 
 - **Memory Mapping (Apache Arrow):**
    - It doesn't load the whole dataset into RAM.
    - It keeps the data on the disk and maps it to memory only when accessed.
 - **Streaming:**
    - You can iterate over the dataset rows instantly without waiting for a full download.
    - `load_dataset('c4', split='train', streaming=True)`.
 
 ---
 
 ## Question 6: TrainingArguments & Trainer
 
 **Topic:** Training
 **Difficulty:** Advanced
 
 ### Question
 Explain the role of the `Trainer` class in Hugging Face.
 
 ### Answer
 
 **Concept:** It abstracts away the training loop (Epochs, Batches, Backward Pass, Optimizer, Scheduler, Logging).
 
 **Components:**
 1. **TrainingArguments:** Hyperparameters (lr, batch_size, epochs, weight_decay, where to save).
 2. **Trainer:** Takes the model, args, train_dataset, eval_dataset, and compute_metrics function.
 3. **Call:** `trainer.train()` runs the entire Fine-Tuning process.
 
 ---
 
 ## Question 7: Pre-trained Model Checkpoints
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What is a "Checkpoint"? What does `model.save_pretrained("./my_model")` actually save?
 
 ### Answer
 
 - **Checkpoint:** A saved state of the model at a specific point in training.
 - **Saved Files:**
    1. **config.json:** Hyperparameters (layers, heads, hidden size).
    2. **vocab.txt / tokenizer.json:** The vocabulary and rules.
    3. **pytorch_model.bin / model.safetensors:** The actual weights (Gigabytes).
 
 *Note: ALWAYS save the tokenizer with the model.*
 
 ---
 
 ## Question 8: RAG (Retrieval Augmented Generation)
 
 **Topic:** Application
 **Difficulty:** Advanced
 
 ### Question
 How would you implement a simple RAG system using Hugging Face and LangChain components?
 
 ### Answer
 
 **Workflow:**
 1. **Ingest:** Load PDF/Text. Split into chunks.
 2. **Embed:** Use a HF model (`sentence-transformers/all-MiniLM-L6-v2`) to vectorise chunks.
 3. **Store:** Put vectors in a Vector DB (FAISS/Chroma).
 4. **Retrieve:** When user asks a question, embed user query -> Search Top-K similar chunks.
 5. **Generate:** Feed `Context + Question` into a Generative LLM (e.g., `Llama-2`).
 
 ---
 
 ## Question 9: Data Collator
 
 **Topic:** Training
 **Difficulty:** Advanced
 
 ### Question
 What does a `DataCollator` do during training? Why is `DataCollatorForLanguageModeling` special?
 
 ### Answer
 
 **Role:**
 - It takes a list of samples (from the dataset) and forms a batch.
 - Handles dynamic padding (pads to the longest in the *current batch*, not max length of model), saving compute.
 
 **For Language Modeling (MLM):**
 - It dynamically **masks** tokens on the fly during training.
 - This means in Epoch 1, the model sees "I [MASK] cat", and in Epoch 2, it might see "I love [MASK]".
 - Acts as a form of Data Augmentation.
 
 ---
 
 ## Question 10: Saving & Loading
 
 **Topic:** Implementation
 **Difficulty:** Basics
 
 ### Question
 Write the code to save a fine-tuned model and then reload it.
 
 ### Answer
 
 ```python
 # SAVE
 directory = "./fine_tuned_bert"
 model.save_pretrained(directory)
 tokenizer.save_pretrained(directory)
 
 # LOAD
 from transformers import AutoModelForSequenceClassification, AutoTokenizer
 
 loaded_model = AutoModelForSequenceClassification.from_pretrained(directory)
 loaded_tok = AutoTokenizer.from_pretrained(directory)
 ```
 
 ---
 
 ## Key Takeaways
 
 - **Hugging Face** is the standard for Open Source NLP.
 - **AutoClasses** (`AutoModel`, `AutoTokenizer`) make code portable.
 - **Trainer API** simplifies the training loop significantly.
 - **Pipelines** are great for inference; **Models** are great for fine-tuning.
 - **Quantization** (BitsAndBytes) allows running huge models on consumer hardware.
 
 **Next:** [Day 22 - Computer Vision](../Day-22/README.md)
