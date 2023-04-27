import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# List of CSV files to plot

ks = [23, 24, 34, 35, 36]
for k in ks:
    try:
        files = [f'/home/mrudolph/documents/fuzzrl/logs/ppo/JPEGEncode-v0_{k}/0.monitor.csv', 
                f'/home/mrudolph/documents/fuzzrl/logs/ppo/JPEGEncode-v0_{k}/1.monitor.csv',
                f'/home/mrudolph/documents/fuzzrl/logs/ppo/JPEGEncode-v0_{k}/2.monitor.csv',
                f'/home/mrudolph/documents/fuzzrl/logs/ppo/JPEGEncode-v0_{k}/3.monitor.csv']

        # Create a figure to plot on
        fig, ax = plt.subplots()
        df_all = pd.DataFrame()
        # Loop over CSV files and plot r vs t on the same figure
        
        results = np.zeros((len(files), 20906))
        results_list = []
        for i, file in enumerate(files):
            # Load CSV file into a pandas dataframe, skipping the first row
            df = pd.read_csv(file, skiprows=1)
            print(df.shape)

            # Add the data to the main dataframe
            results[i] = df['r'].values
            # results_list.append(df['r'].values)
            # df_all = pd.concat([df_all, df])
            


        mean_rew = np.mean(results, axis=0)

        wind = 100
        smooth_rew = np.convolve(mean_rew, np.ones((wind,))/wind, mode='valid')
        plt.plot(smooth_rew)

        # Set the x and y axis labels
        ax.set_xlabel('Eval Epochs')
        ax.set_ylabel('Cumulative Reward')

        # Add a legend to the plot
        # ax.legend()

        # Save the plot to a PNG file
        plt.savefig(f'plots/reward_plot_{k}.png')

        # Show the plot
        plt.show()
        plt.close()
    except:
        pass
