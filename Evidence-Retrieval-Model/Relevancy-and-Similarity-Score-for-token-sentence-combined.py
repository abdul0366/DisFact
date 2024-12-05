import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel, BigBirdTokenizer, BigBirdConfig, RobertaModel, RobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch.multiprocessing.resource_tracker')



class CustomDataset(Dataset):
    def __init__(self, claims, contexts, tokenizer):
        self.claims = claims
        self.contexts = contexts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = self.claims[idx]
        context = self.contexts[idx]

        context_sentences = context if isinstance(context, list) else []
        encoded_claim = self.tokenizer.encode(claim, add_special_tokens=False)
        
        encoded_context = []
        for sentence in context_sentences:
            encoded_context += self.tokenizer.encode(sentence, add_special_tokens=False) + [self.tokenizer.sep_token_id]

        input_ids = [self.tokenizer.cls_token_id] + encoded_claim + [self.tokenizer.sep_token_id] + encoded_context
        attention_mask = [1] * len(input_ids)
        sep_token_indices = [len(encoded_claim) + 1]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "sep_token_indices": sep_token_indices, "context": context_sentences}

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        sep_token_indices = [item["sep_token_indices"] for item in batch]
        contexts = [item["context"] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        for i in range(len(input_ids)):
            padding_length = max_len - len(input_ids[i])
            input_ids[i] += [self.tokenizer.pad_token_id] * padding_length
            attention_mask[i] += [0] * padding_length
            sep_token_indices[i] += [0] * (max_len - len(sep_token_indices[i]))

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "sep_token_indices": torch.tensor(sep_token_indices),
            "contexts": contexts
        }

class SentenceSelectionModel(RobertaModel):
    def __init__(self, model_name, config, device):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bigbird = BigBirdModel.from_pretrained(model_name, config=config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=16)
        self._device = device

    def forward(self, input_ids=None, attention_mask=None, sep_token_indices=None, labels=None, output_hidden_states=True):
        with autocast():
            outputs = self.bigbird(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
            sequence_output = outputs.hidden_states[-1]
            sequence_output, _ = self.multihead_attn(sequence_output, sequence_output, sequence_output)
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()

                loss = loss_fct(logits.view(-1), labels.view(-1).float())

                for i in range(len(attention_mask)):
                    j = sep_token_indices[i].item()
                    attention_mask[i][1:j] = 0

                loss = loss * attention_mask.view(-1).float()
                loss = loss.sum() / attention_mask.sum().float()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save(self, output_path, tokenizer):
        self.bigbird.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        torch.save(self.classifier.state_dict(), os.path.join(output_path, "classification layer"))

    @property
    def device(self):
        return self._device

def map_similarity_to_score(similarity):
    return similarity

def process_claim(claim_idx, claims, contexts, data, model, tokenizer, sentence_transformer):
    if data['label'][claim_idx] == "NOT ENOUGH INFO":
        return claim_idx, None, None, None

    context = contexts[claim_idx]
    if not context:
        return claim_idx, [], [], []

    model_input = tokenizer(claims[claim_idx], context, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_input = {k: v.to(model.device) for k, v in model_input.items()}
        with autocast():
            outputs = model(**model_input)

    hidden_states = outputs.hidden_states[-1].cpu().numpy()

    claim_tokens = tokenizer.tokenize(claims[claim_idx])
    if not claim_tokens:
        return claim_idx, [], [], []
    claim_embeddings = sentence_transformer.encode(claim_tokens, convert_to_tensor=True).cpu().numpy()
    claim_sentence_embedding = sentence_transformer.encode(claims[claim_idx], convert_to_tensor=True).cpu().numpy()

    sentences = context if isinstance(context, list) else []
    if not sentences:
        return claim_idx, [], [], []

    sentence_embeddings = sentence_transformer.encode(sentences, batch_size=16, convert_to_tensor=True).cpu().numpy()

    sentence_level_scores = cosine_similarity(sentence_embeddings, claim_sentence_embedding.reshape(1, -1))
    sentence_level_relevancy_scores = sentence_level_scores.mean(axis=1).tolist()

    token_scores = []
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        if not sentence_tokens:
            token_scores.append([0])
            continue
        token_embeddings = sentence_transformer.encode(sentence_tokens, batch_size=32, convert_to_tensor=True).cpu().numpy()
        scores = cosine_similarity(token_embeddings, claim_embeddings)
        avg_score = scores.mean(axis=1).tolist()
        token_scores.append(avg_score)

    avg_token_scores = [sum(scores) / len(scores) if scores else 0 for scores in token_scores]

    combined_scores = [(token_score + sentence_score) / 2 for token_score, sentence_score in zip(avg_token_scores, sentence_level_relevancy_scores)]
    return claim_idx, avg_token_scores, sentence_level_relevancy_scores, combined_scores

def main():
    input_file_path = ".../data.csv"
    data = pd.read_csv(input_file_path, encoding='utf-8')
    # data = data.head()

    data['token_level_scores'] = None
    data['sentence_level_scores'] = None
    data['combined_level_scores'] = None

    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    claims = data['claim'].tolist()
    contexts = data['context'].apply(lambda x: eval(x) if isinstance(x, str) else []).tolist()

    config = BigBirdConfig.from_pretrained('google/bigbird-roberta-base')
    config.attention_type = "original_full"
    config.relative_attention_num_buckets = 1000
    config.output_hidden_states = True
    model = SentenceSelectionModel('google/bigbird-roberta-base', config, device="cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)

    data_dict = {
        'claims': claims,
        'contexts': contexts,
        'data': data,
        'model': model,
        'tokenizer': tokenizer,
        'sentence_transformer': sentence_transformer
    }

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_claim, claim_idx, **data_dict): claim_idx for claim_idx in range(len(claims))}
        
        processed_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"Processed {processed_count} claims")

    for result in results:
        claim_idx, token_level_scores, sentence_level_scores, combined_level_scores = result
        if claim_idx is not None:
            data.at[claim_idx, 'token_level_scores'] = token_level_scores
            data.at[claim_idx, 'sentence_level_scores'] = sentence_level_scores
            data.at[claim_idx, 'combined_level_scores'] = combined_level_scores

    output_file_path = "..../relevancy_with_scores.csv"
    
    # Check available space before writing
    total, used, free = shutil.disk_usage("/")
    print("Disk space available: ", free // (2**30), "GB")

    if free // (2**30) < 1:  # If less than 1 GB is available
        print("Warning: Disk space is critically low. Unable to save the file.")
    else:
        data.to_csv(output_file_path, index=False)
        print(f"Results have been saved to {output_file_path}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
