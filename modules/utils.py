from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

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