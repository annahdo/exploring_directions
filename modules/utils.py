from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
import os

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# plot functions
def plot_lines(data_dict, y_name, save_path=None, method_names=None, loc='upper left'):
    # plot
    data_list = [{'Method': method, 'Layer': layer, y_name: value} 
                for method, layers in data_dict.items() 
                for layer, value in layers.items()]

    # Convert to DataFrame
    df = pd.DataFrame(data_list)

    # Pivot DataFrame
    df_pivot = df.pivot(index='Layer', columns='Method', values=y_name)

    if method_names:
        df_pivot = df_pivot[method_names]
    # Plot
    df_pivot.plot(kind='line', marker='o', figsize=(8, 5))

    plt.xlabel('Layer')
    plt.ylabel(y_name)
    plt.grid(True)
    plt.legend(title='Method', loc=loc)
    if save_path:
        plt.savefig(save_path)

    plt.show()

def load_util_data(data_dir, split="test"):
    data = pd.read_csv(data_dir + f'util_{split}.csv', header=None)
    sentences = []
    for d in data.iloc:
        sentences.append([d[0], d[1]])
    return np.array(sentences)


def mix_util_pairs(data, random_seed=0):
    # per default always the first sentence is the sentence with higher utility
    # this function swaps the sentences with probability 0.5
    rng = np.random.default_rng(random_seed)
    m = len(data)
    labels = rng.integers(0, 2, size=m)
    data = data[np.arange(m), [1-labels, labels]].T
    return data, labels

def find_two_sentences(data, split_str1=".", split_str2=",", larger_than1=2, larger_than2=1):
    idxs = []
    sentences = []

    for i, d in enumerate(data):
        if len(d.split(split_str1)) > larger_than1:
            idxs.append(i)
            sentences.append(d.split(split_str1)[0] + f"{split_str1}")
        elif split_str2:
            if len(d.split(split_str2)) > larger_than2:
                sentences.append(d.split(split_str2)[0]+f"{split_str2}")
                idxs.append(i)
    return idxs, sentences

def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None] 
    return logprobs.squeeze(-1)

def get_logits(tokenizer, model, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model(input_ids, attention_mask=attention_mask).logits

    return output, attention_mask, input_ids

def eval_perplexity(sentences, batch_size, tokenizer, wrapped_model, device):
    perplexities = []
    for sentence_batch in batchify(sentences, batch_size):
        logits, attention_mask, input_ids = get_logits(tokenizer, wrapped_model, sentence_batch, device)
        p = get_logprobs(logits, input_ids, attention_mask)
        perplexities.extend(list(torch.exp(-p.mean(dim=-1)).detach().cpu().float().numpy()))

    return np.mean(perplexities)

def eval_sentiment(generations, batch_size, tokenizer, sentiment_model, device):
        outputs = {"positive": [], "negative": []}
        # eval
        for sentence_batch in batchify(generations["positive"], batch_size):
            logits, _, _ = get_logits(tokenizer, sentiment_model, sentence_batch, device)
            output = softmax(logits.detach().float().cpu().numpy(), axis=-1)
            outputs["positive"].append(output)

        outputs["positive"] = np.concatenate(outputs["positive"], axis=0)

        for sentence_batch in batchify(generations["negative"], batch_size):
            logits, _, _ = get_logits(tokenizer, sentiment_model, sentence_batch, device)
            output = softmax(logits.detach().float().cpu().numpy(), axis=-1)
            outputs["negative"].append(output)

        outputs["negative"] = np.concatenate(outputs["negative"], axis=0)

        # output has three values per sample: negative, neutral, positive
        # calculate accuracy
        pos_bigger = (outputs["positive"][:,-1]>outputs["negative"][:,-1])

        return np.mean(pos_bigger)

def load_generations(file_name):
    with open(file_name, "r") as f:
        generations = [line.strip() for line in f]
    return generations