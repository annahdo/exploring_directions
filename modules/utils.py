from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# plot functions
def plot_lines(data_dict, y_name, save_path=None, method_names=None):
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
    plt.legend(title='Method')
    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_bars(data_dict, y_name, save_path=None):
    data_list = []

    for method, layers in data_dict.items():
        for layer, value in layers.items():
            data_list.append({'Method': method, 'Layer': layer, y_name: value})

    df = pd.DataFrame(data_list)

    # Initialize the figure
    plt.figure(figsize=(8, 5))

    # Create a barplot for median
    #sns.barplot(x='Method', y='Accuracy', data=df, estimator=np.median, ci=None, color='grey', alpha=0.6, label='Median')

    # Create a boxplot for quartiles etc.
    sns.barplot(x='Method', y=y_name, width=0.3, data=df)
    # Labels and title
    plt.xlabel('Method')
    plt.ylabel(y_name)
    plt.grid(linestyle='-', linewidth=0.7, alpha=0.7)    
    plt.xticks(rotation=45)

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

def load_generations(file_name):
    with open(file_name, "r") as f:
        generations = [line.strip() for line in f]
    return generations