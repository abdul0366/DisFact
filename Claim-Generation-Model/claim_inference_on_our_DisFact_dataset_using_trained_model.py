import pandas as pd
import random
import nltk
from simplet5 import SimpleT5
from transformers import T5Tokenizer
from tqdm import tqdm  # Import tqdm for the progress bar

# Ensure you have the punkt tokenizer
# nltk.download('punkt')

# Set the random seed
random.seed(42)

# Initialize models
support_model_path = ".../outputs-support/simplet5-epoch-49-train"
refute_model_path = ".../outputs-refute/simplet5-epoch-49-train"

supportModel = SimpleT5()
supportModel.load_model("t5", support_model_path, use_gpu=True)

refuteModel = SimpleT5()
refuteModel.load_model("t5", refute_model_path, use_gpu=True)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
MAX_LEN = 512  # Maximum length for T5

GENERATED_MAX_LEN = 50

# Define functions to generate supported and refuted claims in batches
def genSupportClaims(texts):
    tokenized_texts = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    tokenized_texts = {key: value.cuda() for key, value in tokenized_texts.items()}  # Move to GPU
    generated_ids = supportModel.model.generate(**tokenized_texts, max_length=GENERATED_MAX_LEN)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

def genRefuteClaims(texts):
    tokenized_texts = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    tokenized_texts = {key: value.cuda() for key, value in tokenized_texts.items()}  # Move to GPU
    generated_ids = refuteModel.model.generate(**tokenized_texts, max_length=GENERATED_MAX_LEN)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

# Load the data
path = "new_and_old_fema.csv"
df = pd.read_csv(path)

# Process each article
claims_data = []
batch_size = 64  # Define an appropriate batch size based on your GPU memory
batch_sentences = []
batch_info = []  # Store information about each sentence in the batch

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing articles"):
    text = row['Text']
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) < 2:
        continue  # Skip articles with less than 2 sentences

    # Randomly select two sentences
    selected_sentences = random.sample(sentences, 2)
    
    for sentence in selected_sentences:
        modified_sentence = f"{sentence} (Article title: {row['Title']})"
        batch_sentences.append(modified_sentence)
        batch_info.append({
            "Article Text": text,
            "Article Title": row['Title'],
            "Date": row['Date'],
            "Evidence Sentence": modified_sentence
        })


    if len(batch_sentences) >= batch_size:
        # Process the batch
        support_claims = genSupportClaims(batch_sentences)
        refute_claims = genRefuteClaims(batch_sentences)

        for info, support_claim, refute_claim in zip(batch_info, support_claims, refute_claims):
            claims_data.append({
                "Claim": support_claim,
                "Evidence Sentence": info["Evidence Sentence"],
                "Article Text": info["Article Text"],
                "Article Title": info["Article Title"],
                "Date": info["Date"],
                "Label": "Supported"
            })
            claims_data.append({
                "Claim": refute_claim,
                "Evidence Sentence": info["Evidence Sentence"],
                "Article Text": info["Article Text"],
                "Article Title": info["Article Title"],
                "Date": info["Date"],
                "Label": "Refuted"
            })

        batch_sentences = []  # Clear the batch after processing
        batch_info = []       # Clear the info batch after processing

# Process any remaining sentences in the last batch
if batch_sentences:
    support_claims = genSupportClaims(batch_sentences)
    refute_claims = genRefuteClaims(batch_sentences)

    for info, support_claim, refute_claim in zip(batch_info, support_claims, refute_claims):
        claims_data.append({
            "Claim": support_claim,
            "Evidence Sentence": info["Evidence Sentence"],
            "Article Text": info["Article Text"],
            "Article Title": info["Article Title"],
            "Date": info["Date"],
            "Label": "Supported"
        })
        claims_data.append({
            "Claim": refute_claim,
            "Evidence Sentence": info["Evidence Sentence"],
            "Article Text": info["Article Text"],
            "Article Title": info["Article Title"],
            "Date": info["Date"],
            "Label": "Refuted"
        })

# Save the claims to a new CSV file
output_path = "sentence_level_fema_claims_from_title_v3.csv"
claims_df = pd.DataFrame(claims_data)
claims_df.to_csv(output_path, index=False)

print(f"Claims saved to {output_path}")
