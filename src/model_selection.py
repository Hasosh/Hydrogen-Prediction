import pandas as pd 
import wandb
import matplotlib.pyplot as plt
import os

# Config File
from config import Config


# Define a function to create and show horizontal bar plots
def create_bar_plot(group, metric, title, ascending):
    sorted_group = group.sort_values(by=metric, ascending=ascending)
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_group['name'], sorted_group[metric])
    if metric == "avg_cosine_similarity":
        plt.xlabel("avg_error_angle")
    else:
        plt.xlabel(metric)
    plt.ylabel('Model Name')
    train_data_path = group["train_data_path"].iloc[0]

    # Ensure the directory exists
    directory = f"../img/training/{train_data_path}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.title(f'{title} for {train_data_path}')
    plt.savefig(f"{directory}/{metric}.png", bbox_inches='tight')
    plt.show()

# Function to plot rankings for each dataset
def plot_rankings(rankings):
    for dataset, ranking_list in rankings.items():
        names, avg_ranks = zip(*ranking_list)
        plt.figure(figsize=(12, 8))
        plt.barh(names, avg_ranks, color='skyblue')
        plt.xlabel('Average Rank')
        plt.ylabel('Model Name')
        plt.title(f'Average Rank of Models for {dataset}')
        plt.gca().invert_yaxis()  # Invert y-axis to have the best rank at the top
        plt.tight_layout()

        # Ensure the directory exists
        directory = f"../img/training/{dataset}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the plot with a bounding box
        plt.savefig(f"{directory}/ranking.png", bbox_inches='tight')
        plt.show()


if __name__=="__main__":
    # -------------------------------------------------
    # Get training results from wandb and process
    # -------------------------------------------------

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{Config.wandb_entity}/{Config.wandb_project_name}")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
    
    # Create DataFrames from the summary and config lists
    summary_df = pd.json_normalize(summary_list)
    config_df = pd.json_normalize(config_list)

    # Combine the summary, config, and name DataFrames
    runs_df = pd.concat([summary_df, config_df], axis=1)
    runs_df['name'] = name_list

    # -------------------------------------------------
    # Model Metric Plots per Dataset
    # -------------------------------------------------

    # Group the DataFrame by 'train_data_path'
    grouped = runs_df.groupby('train_data_path')

    # For each dataset group, create the specified plots
    for train_data_path, group in grouped:
        if train_data_path == "dataset-C4-depth1":
            group = group[group['name'] != 'polynomial_linear_regression_order_2']
            
        print(f"Creating plots for dataset: {train_data_path}")
        create_bar_plot(group, 'MSE', 'MSE (Lowest to Highest)', ascending=False)
        create_bar_plot(group, 'R2', 'R2 (Highest to Lowest)', ascending=True)
        create_bar_plot(group, 'avg_binding_length', 'Avg Binding Length (Lowest to Highest)', ascending=False)
        create_bar_plot(group, 'avg_cosine_similarity', 'Avg Error Angle (Lowest to Highest)', ascending=False)
        create_bar_plot(group, 'tri-angle_wasserstein_distance', 'Tri-angle Wasserstein Distance (Lowest to Highest)', ascending=False)

    # -------------------------------------------------
    # Model Ranking Plots per Dataset
    # -------------------------------------------------

    # Initialize a dictionary to store rankings per dataset
    rankings = {}

    # For each dataset group, create the specified plots and calculate ranks
    best_models = {}

    for train_data_path, group in grouped:
        # Rank models for each metric
        group['rank_MSE'] = group['MSE'].rank(ascending=True)
        group['rank_R2'] = group['R2'].rank(ascending=False)
        group['rank_avg_binding_length'] = group['avg_binding_length'].rank(ascending=True)
        group['rank_avg_cosine_similarity'] = group['avg_cosine_similarity'].rank(ascending=True)
        group['rank_tri_angle_wasserstein_distance'] = group['tri-angle_wasserstein_distance'].rank(ascending=True)
        
        # Calculate average rank
        group['average_rank'] = group[['rank_MSE', 'rank_R2', 'rank_avg_binding_length', 'rank_avg_cosine_similarity', 'rank_tri_angle_wasserstein_distance']].mean(axis=1)

        # Create a list of (name, average rank) tuples
        ranking_list = list(zip(group['name'], group['average_rank']))

        # Sort the list by average rank
        ranking_list.sort(key=lambda x: x[1])

        # Store the ranking list in the rankings dictionary
        rankings[train_data_path] = ranking_list

        # Identify the best model
        best_model = group.loc[group['average_rank'].idxmin()]
        best_models[train_data_path] = best_model['name']

        print(f"Best model for {train_data_path}: {best_model['name']} with average rank {best_model['average_rank']}")

    # Call the function to plot the rankings
    plot_rankings(rankings)