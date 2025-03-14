import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import sentencepiece
print(sentencepiece.__version__)

# 1. Data Preparation
def prepare_translation_data(file_paths):
    """
    Combines and preprocesses translation data from given file paths.

    Args:
        file_paths (list): A list of file paths with naming convention 'src_tgt.txt' (e.g., 'en_hi.txt').

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data with source and target language tokens.
    """
    combined_data = []
    for file_path in file_paths:
        print(file_path.split("/")[-1].split(".")[0].split("_"))
        src_lang, tgt_lang = file_path.split("/")[-1].split(".")[0].split("_")  # Extract language codes from filename

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    src_text, tgt_text = line.strip().split('\t')  # Assuming tab-separated data
                    combined_data.append({
                        'source_text': f"<{src_lang}> {src_text}",
                        'target_text': f"<{tgt_lang}> {tgt_text}"
                    })
                    #for reverse translation
                    combined_data.append({
                        'source_text': f"<{tgt_lang}> {tgt_text}",
                        'target_text': f"<{src_lang}> {src_text}"
                    })
                except ValueError:
                    print(f"Skipping line due to formatting issue: {line.strip()} in {file_path}") #log if any formatting issues

    return pd.DataFrame(combined_data)

# 2. Fine-tuning the MT5 Model
def fine_tune_mt5(model_name, train_df, output_dir, batch_size=8, epochs=3, max_length=128): # Added max_length
    """
    Fine-tunes the MT5 model on the provided training data.

    Args:
        model_name (str): Name or path of the pre-trained MT5 model.
        train_df (pandas.DataFrame): DataFrame containing training data with 'source_text' and 'target_text' columns.
        output_dir (str): Directory to save the fine-tuned model.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        epochs (int, optional): Number of training epochs. Defaults to 3.
        max_length (int, optional): Maximum length for tokenization. Defaults to 128.
    """

    tokenizer = MT5Tokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['source_text'], max_length=max_length, truncation=True, padding='max_length') # Added padding
        labels = tokenizer(examples['target_text'], max_length=max_length, truncation=True, padding='max_length') # Added padding
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = train_df.to_dict(orient='list')
    tokenized_datasets = tokenize_function(tokenized_datasets)
    
    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized_data):
            self.tokenized_data = tokenized_data

        def __len__(self):
            return len(self.tokenized_data['input_ids'])

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in enumerate(self.tokenized_data['input_ids'])} # Changed how items are accessed

    train_dataset = TranslationDataset(tokenized_datasets)

    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 3. Inference (Translation)
def translate_text(model_dir, text, src_lang, tgt_lang, max_length=128): # Added max_length
    """
    Translates the given text from source language to target language using the fine-tuned MT5 model.

    Args:
        model_dir (str): Directory containing the fine-tuned MT5 model.
        text (str): Text to translate.
        src_lang (str): Source language code (e.g., 'en').
        tgt_lang (str): Target language code (e.g., 'hi').
        max_length (int, optional): Maximum length for tokenization. Defaults to 128.

    Returns:
        str: Translated text.
    """

    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)

    input_text = f"<{src_lang}> {text} <extra_id_0> <{tgt_lang}>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length') # Added max_length, padding and truncation

    output_ids = model.generate(input_ids, max_length=max_length)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text

# Main Execution
if __name__ == "__main__":
    # Define your file paths
    file_paths = [
        "en_bn.tsv",
        "en_hi.tsv",
        "en_kn.tsv",
        "en_mr.tsv",
        "en_ta.tsv",
        "en_te.tsv",
        # Add paths to your other language pair files here (e.g., "en_es.txt", "en_fr.txt")
    ]
    
    # Prepare the data
    train_df = prepare_translation_data(file_paths)
    print("Prepared Data sample:")
    print(train_df.head())

    # Fine-tune the model
    model_name = "google/mt5-small"  # Or any other MT5 variant
    output_dir = "fine_tuned_mt5_model"
    fine_tune_mt5(model_name, train_df, output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

    # Inference example
    model_dir = output_dir  # Use the directory where you saved the fine-tuned model
    text_to_translate = "This is a sample sentence."
    source_language = "en"
    target_language = "mni"
    
    translated_text = translate_text(model_dir, text_to_translate, source_language, target_language)
    print(f"Translation from {source_language} to {target_language}:")
    print(f"Input: {text_to_translate}")
    print(f"Output: {translated_text}")