import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score
import warnings
from datasets import load_dataset




def custom_formatwarning(message, category, filename, lineno, line=None):
    # Customize the warning message to only include the category and the message.
    return f"{category.__name__}: {message}\n"

# Override the default warning formatter.
warnings.formatwarning = custom_formatwarning


def get_movie_answer_dictionaries(dataset_name, results_path=None):
    """
    Loads the dataset using the provided dataset name and split, then processes it to return two dictionaries that map movies to their associated answers, but only for movies that have corresponding folders in the results_path (if provided).
    
    The provided dataset is expected to have columns 'Movie', 'Label', and 'Answer'. 
    It groups by 'Movie' (assuming the answer for each movie is unique) and returns:
      - a dictionary for rows with Label 0 (clean),
      - a dictionary for rows with Label 1 (suspect).
    
    Additionally, if results_path is provided, the dictionaries are filtered to only include movies whose names appear as directories in that folder.
    
    Furthermore, if there are folders in the results_path that do not correspond to any movie in the dataset (either clean or suspect), a warning is raised.
    
    Parameters:
        dataset_name (str): Name or identifier of the dataset (e.g., "DIS-CO/MovieTection_Mini").
        results_path (str, optional): Path to the folder containing movie result directories.
                                      Only movies with a corresponding directory will be included.
    
    Returns:
        tuple: (movie_list_clean, movie_list_suspect) where:
            - movie_list_clean is a dict mapping movies to their answer for Label 0,
            - movie_list_suspect is a dict mapping movies to their answer for Label 1.
    """

    # Load the dataset and keep only the 'Movie', 'Label', and 'Answer' columns.
    print("\nLoading Dataset to Create Movie Answer Dictionaries...")
    dataset = load_dataset(dataset_name, split="train")
    df = dataset.to_pandas()[['Movie', 'Label', 'Answer']]
    print("Dataset Loaded Successfully.\n")
    
    # Create a dictionary to hold the results for each label.
    label_dictionaries = {}

    # Process each unique label.
    for label in df['Label'].unique():
        # Filter the DataFrame for the current label.
        df_label = df[df['Label'] == label]
        
        # Group by 'Movie' and take the first entry of 'Answer'.
        # Convert the result to a Python list if it is a NumPy array.
        movie_answer_dict = df_label.groupby('Movie')['Answer'].first() \
            .apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x) \
            .to_dict()
        
        # Store the dictionary for this label.
        label_dictionaries[label] = movie_answer_dict

    # Extract the dictionaries for labels 0 and 1 (or return empty dictionaries if not found).
    movie_list_clean = label_dictionaries.get(0, {})
    movie_list_suspect = label_dictionaries.get(1, {})
    
    # If a results_path is provided, filter the dictionaries to only include movies that exist as directories.
    if results_path is not None:
        if os.path.exists(results_path) and os.path.isdir(results_path):
            # Get list of directory names in the results_path.
            available_movies = {
                name for name in os.listdir(results_path) 
                if os.path.isdir(os.path.join(results_path, name))
            }
            
            # Determine the set of movies that are in the dataset.
            dataset_movies = set(movie_list_clean.keys()) | set(movie_list_suspect.keys())
            
            # For each folder that doesn't match any movie in the dataset, raise a warning.
            for movie in available_movies - dataset_movies:
                warnings.warn(f"Movie: {movie} is not found in the original dataset {dataset_name}")
            
            # Filter each dictionary to only include movies present in the available_movies set.
            movie_list_clean = {
                movie: answer for movie, answer in movie_list_clean.items() 
                if movie in available_movies
            }
            movie_list_suspect = {
                movie: answer for movie, answer in movie_list_suspect.items() 
                if movie in available_movies
            }
        else:
            raise ValueError(f"The results_path provided ({results_path}) is not a valid directory.")
    else:
        raise ValueError(f"The results_path directory is not provided.")
    
    return movie_list_clean, movie_list_suspect


def calculate_overall_accuracy(df, gold_answer, cols_of_interest):
    """
    Calculate the movie overall accuracy.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        gold_answer (list): List of strings representing the correct answers.
        cols_of_interest (list): List of column names to include in the calculation.

    Returns:
        float: Overall accuracy as a percentage.
    """
    # Check if each value in the specified columns is in the gold_answer list
    correct = df[cols_of_interest].map(lambda x: x in gold_answer)
    total_correct = correct.sum().sum()  # Sum across both rows and columns
    total_elements = correct.size  # Total number of elements in the selected columns
    return total_correct / total_elements  # Return percentage accuracy


def build_final_output(df_main_accuracies, df_neutral_accuracies):
    """
    Takes the dataframe results containing the main and neutral accuracies and converts them into a nested dictionary with two main sections:

      1) 'Grouped Results': 
         For each model, provides the average main, and neutral accuracies across all movies.
         If any movie has a NaN value for that metric, the grouped result is NaN.

      2) 'Movie-Level Results': 
         For each model, provides a dictionary of each movie’s main, and neutral accuracies.

    Args:
        df_main_accuracies (dict):
            {model_name: DataFrame} containing main accuracies for each movie in a single column named after the model.

        df_neutral_accuracies (dict):
            {model_name: DataFrame} containing neutral accuracies for each movie.

    Returns:
        dict: 
            A nested dictionary with structure:
            {
              "Grouped Results": {
                modelName: {"main": float, "neutral": float},
                ...
              },
              "Movie-Level Results": {
                modelName: {
                  movieName: {"main": float, "neutral": float},
                  ...
                },
                ...
              }
            }
    """
    final_dict = {
        'Grouped Results': {},
        'Movie-Level Results': {}
    }

    # For each model, we'll retrieve the 2 DataFrames (main, neutral)
    for model in df_main_accuracies.keys():
        df_main = df_main_accuracies[model]
        df_neutral = df_neutral_accuracies[model]

        # -- GROUPED RESULTS (mean accuracy across all movies, with skipna=False)
        # This means if ANY value is NaN, the mean becomes NaN too.
        mean_main = df_main[model].mean(skipna=False)
        mean_neutral = df_neutral[model].mean(skipna=False)

        final_dict['Grouped Results'][model] = {
            'main': float(mean_main),
            'neutral': float(mean_neutral),
        }

        # -- MOVIE-LEVEL RESULTS (one entry per movie)
        model_movie_dict = {}
        for movie_name in df_main.index:
            model_movie_dict[movie_name] = {
                'main': float(df_main.loc[movie_name, model]),
                'neutral': float(df_neutral.loc[movie_name, model]),
            }

        final_dict['Movie-Level Results'][model] = model_movie_dict

    return final_dict


def process_movie_data(movie_list, models, path, method):
    """
    Processes movie data (i.e. calculate accuracies) for one of three possible methods:
      1) 'disco':
         - Uses single image files (_single_image.xlsx).

      2) 'captions':
         - Uses single caption files (_single_caption.xlsx).

      3) 'disco_floor':
         - Loads both image and caption files.
         - Aligns them (ensuring they have the same shape) and replaces matching cells with the placeholder 'XX-REMOVED-XX'
           in the image DataFrame before calculating accuracy. 
         - Provides a more strict classification of possible suspect movies as it only considers frames that were not correctly
           classified by the caption data.


    Args:
        movie_list (dict):
            A dictionary mapping each movie's name to its gold (correct) answer.

        models (list):
            A list of model names (strings) which are used as the file-part needed to locate that model's Excel files.

        path (str):
            The base directory where the files for each movie are stored.

        method (str):
            One of {'disco', 'captions', 'disco_floor'}

    Returns:
        dict:
            A nested dictionary with two main keys: "Grouped Results" and "Movie-Level Results".
    """

    # We return a structure with NaN for all movies/models if someone provides an invalid method
    if method not in ['disco', 'captions', 'disco_floor']:
        warnings.warn("Invalid method specified. Returning NaN results for all movies.")
        
        # Build a list of all movie names
        movie_names = list(movie_list.keys())
        
        # For each model, we create a DataFrame with NaN for each movie
        df_main_accuracies = {
            model: pd.DataFrame({model: [np.nan]*len(movie_names)}, index=movie_names)
            for model in models
        }
        df_neutral_accuracies = {
            model: pd.DataFrame({model: [np.nan]*len(movie_names)}, index=movie_names)
            for model in models
        }

        # Convert these DataFrames into the final nested structure
        return build_final_output(df_main_accuracies, df_neutral_accuracies)

    # Based on the method, define which columns to look at and the file suffix for main vs. neutral data. 
    if method == 'disco':
        main_cols = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
        neutral_cols = ['Image 1', 'Image 2', 'Image 3', 'Image 4']
        file_suffix = '_single_image.xlsx'

    elif method == 'captions':
        main_cols = ['Caption 1', 'Caption 2', 'Caption 3', 'Caption 4', 'Caption 5']
        neutral_cols = ['Caption 1', 'Caption 2', 'Caption 3', 'Caption 4']
        file_suffix = '_single_caption.xlsx'

    elif method == 'disco_floor':
        main_cols_image = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
        neutral_cols_image = ['Image 1', 'Image 2', 'Image 3', 'Image 4']
        file_suffix_image = '_single_image.xlsx'
        file_suffix_caption = '_single_caption.xlsx'

    # We'll store lists of accuracies in dictionaries keyed by model.
    movie_names = []
    main_accuracies = {model: [] for model in models}
    neutral_accuracies = {model: [] for model in models}

    # -------------------------------------------------------------------------
    #  Load data movie by movie, model by model, and compute main/neutral accuracies. 
    #  For 'disco_floor', we do the merge step.
    # -------------------------------------------------------------------------
    for movie_name, gold_answer in movie_list.items():
        # Keep track of each movie's name in the same order
        movie_names.append(movie_name)

        for model in models:
            
            # Initialize accuracy placeholders for this model & movie
            main_acc = np.nan
            neutral_acc = np.nan

            # -----------------------------------------------------------------
            # For 'disco' or 'captions'
            # -----------------------------------------------------------------
            if method in ['disco', 'captions']:
                main_file = f"{path}/{movie_name}/{movie_name}_results_main_{model}{file_suffix}"
                neutral_file = f"{path}/{movie_name}/{movie_name}_results_neutral_{model}{file_suffix}"

                if os.path.exists(main_file):
                    try:
                        df_main = pd.read_excel(main_file)
                        main_acc = calculate_overall_accuracy(df_main, gold_answer, main_cols)
                    except Exception:
                        warnings.warn(f"Error processing main file for {model} in {movie_name}.")
                else:
                    warnings.warn(f"main file missing for {model} in {movie_name}.")

                if os.path.exists(neutral_file):
                    try:
                        df_neutral = pd.read_excel(neutral_file)
                        neutral_acc = calculate_overall_accuracy(df_neutral, gold_answer, neutral_cols)
                    except Exception:
                        warnings.warn(f"Error processing neutral file for {model} in {movie_name}.")
                else:
                    warnings.warn(f"neutral file missing for {model} in {movie_name}.")

            # -----------------------------------------------------------------
            # For 'disco_floor', load both image and caption versions, merge them
            # -----------------------------------------------------------------
            elif method == 'disco_floor':
                main_file_image = f"{path}/{movie_name}/{movie_name}_results_main_{model}{file_suffix_image}"
                main_file_caption = f"{path}/{movie_name}/{movie_name}_results_main_{model}{file_suffix_caption}"

                neutral_file_image = f"{path}/{movie_name}/{movie_name}_results_neutral_{model}{file_suffix_image}"
                neutral_file_caption = f"{path}/{movie_name}/{movie_name}_results_neutral_{model}{file_suffix_caption}"

                # 1) Load & merge main
                if os.path.exists(main_file_image) and os.path.exists(main_file_caption):
                    try:
                        df_main_img = pd.read_excel(main_file_image)
                        df_main_cap = pd.read_excel(main_file_caption)
                    except Exception:
                        warnings.warn(f"Error reading main files for {model} in {movie_name}.")
                        df_main_img = None
                        df_main_cap = None

                    if df_main_img is not None and df_main_cap is not None:
                        if df_main_img.shape == df_main_cap.shape:
                            df_main_cap_aligned = df_main_cap.copy()
                            df_main_cap_aligned.columns = df_main_img.columns

                            df_main_combined = df_main_img.copy()

                            comparison_columns = [col for col in df_main_img.columns if col != 'Scene']
                            df_img_str = df_main_img[comparison_columns].astype(str)
                            df_cap_str = df_main_cap_aligned[comparison_columns].astype(str)

                            # Mark identical cells
                            for row in range(df_img_str.shape[0]):
                                for col in comparison_columns:
                                    if df_img_str.iloc[row][col] == df_cap_str.iloc[row][col]:
                                        df_main_combined.at[row, col] = 'XX-REMOVED-XX'

                            main_acc = calculate_overall_accuracy(df_main_combined, gold_answer, main_cols_image)
                        else:
                            warnings.warn(f"Dimension mismatch in main files for {model} in {movie_name}.")
                else:
                    if not os.path.exists(main_file_image):
                        warnings.warn(f"main image file missing for {model} in {movie_name}.")
                    if not os.path.exists(main_file_caption):
                        warnings.warn(f"main caption file missing for {model} in {movie_name}.")

                # 2) Load & merge neutral
                if os.path.exists(neutral_file_image) and os.path.exists(neutral_file_caption):
                    try:
                        df_neutral_img = pd.read_excel(neutral_file_image)
                        df_neutral_cap = pd.read_excel(neutral_file_caption)
                    except Exception:
                        warnings.warn(f"Error reading neutral files for {model} in {movie_name}.")
                        df_neutral_img = None
                        df_neutral_cap = None

                    if df_neutral_img is not None and df_neutral_cap is not None:
                        if df_neutral_img.shape == df_neutral_cap.shape:
                            df_neutral_cap_aligned = df_neutral_cap.copy()
                            df_neutral_cap_aligned.columns = df_neutral_img.columns
                            
                            df_neutral_combined = df_neutral_img.copy()
                            comparison_columns = [col for col in df_neutral_img.columns if col != 'Scene']
                            df_neutral_img_str = df_neutral_img[comparison_columns].astype(str)
                            df_neutral_cap_str = df_neutral_cap_aligned[comparison_columns].astype(str)

                            for row in range(df_neutral_img_str.shape[0]):
                                for col in comparison_columns:
                                    if df_neutral_img_str.iloc[row][col] == df_neutral_cap_str.iloc[row][col]:
                                        df_neutral_combined.at[row, col] = 'XX-REMOVED-XX'

                            neutral_acc = calculate_overall_accuracy(df_neutral_combined, gold_answer, neutral_cols_image)
                        else:
                            warnings.warn(f"Dimension mismatch in neutral files for {model} in {movie_name}.")
                else:
                    if not os.path.exists(neutral_file_image):
                        warnings.warn(f"neutral image file missing for {model} in {movie_name}.")
                    if not os.path.exists(neutral_file_caption):
                        warnings.warn(f"neutral caption file missing for {model} in {movie_name}.")

            # Store the computed accuracies
            main_accuracies[model].append(main_acc)
            neutral_accuracies[model].append(neutral_acc)

    # -------------------------------------------------------------------------
    #  Convert to DataFrames
    # -------------------------------------------------------------------------
    df_main_accuracies = {}
    df_neutral_accuracies = {}

    for model in models:
        df_main_accuracies[model] = pd.DataFrame({model: main_accuracies[model]}, index=movie_names)
        df_neutral_accuracies[model] = pd.DataFrame({model: neutral_accuracies[model]}, index=movie_names)
        
    # -------------------------------------------------------------------------
    #  Build final output 
    # -------------------------------------------------------------------------
    final_output = build_final_output(df_main_accuracies, df_neutral_accuracies)
    return final_output


def compute_auc(results_clean, results_suspect):
    """
    Compute AUC for 'main' and 'neutral' accuracies given two dictionaries returned by process_movie_data(...):
      - results_clean: data for 'clean' movies
      - results_suspect: data for 'suspect' movies

    Each of these dictionaries has the shape:
        {
          "Grouped Results": {...},
          "Movie-Level Results": {
             modelName: {
                movieName: {"main": val, "neutral": val},
                ...
             },
             ...
          }
        }

    We'll compute:
      - AUC_main
      - AUC_neutral

    If a ValueError (e.g., NaN data) is encountered while computing AUC,
    the model is still kept in the final results but assigned NaN for whichever metric(s)
    caused the error.

    Returns:
        dict:
          {
            model_name: {
              "AUC_main": <float or NaN>,
              "AUC_neutral": <float or NaN>,
            },
            ...
          }
    """
    auc_scores = {}

    ml_clean = results_clean["Movie-Level Results"]
    ml_suspect = results_suspect["Movie-Level Results"]

    for model in ml_clean.keys():
        # Gather data
        main_clean_vals = []
        neutral_clean_vals = []
        main_suspect_vals = []
        neutral_suspect_vals = []

        # For clean
        for movie_name, stats in ml_clean[model].items():
            main_clean_vals.append(stats["main"])
            neutral_clean_vals.append(stats["neutral"])

        # For suspect
        for movie_name, stats in ml_suspect[model].items():
            main_suspect_vals.append(stats["main"])
            neutral_suspect_vals.append(stats["neutral"])

        main_clean_vals = np.array(main_clean_vals)
        neutral_clean_vals = np.array(neutral_clean_vals)
        main_suspect_vals = np.array(main_suspect_vals)
        neutral_suspect_vals = np.array(neutral_suspect_vals)

        y_main = np.concatenate([main_clean_vals, main_suspect_vals])
        y_neutral = np.concatenate([neutral_clean_vals, neutral_suspect_vals])

        # 0 for clean, 1 for suspect
        labels = np.array([0]*len(main_clean_vals) + [1]*len(main_suspect_vals))

        auc_main = np.nan
        auc_neutral = np.nan

        try:
            auc_main = roc_auc_score(labels, y_main)
        except ValueError as e:
            warnings.warn(f"Could not compute AUC_main for model {model}: {str(e)}")

        try:
            auc_neutral = roc_auc_score(labels, y_neutral)
        except ValueError as e:
            warnings.warn(f"Could not compute AUC_neutral for model {model}: {str(e)}")

        auc_scores[model] = {
            "AUC_main": auc_main,
            "AUC_neutral": auc_neutral
        }

    return auc_scores


class Metrics:

    def __init__(
        self,
        method,
        models,
        dataset_name,
        results_base_folder,
        metrics_output_directory
    ):
        """
        A class to run the complete metrics computation pipeline.

        Args:
            method (str): "disco", "captions", or "disco_floor"
            models (list): List of model names (strings) e.g. ['gpt-4o-2024-08-06', 'gemini-1.5-pro', ...]
            dataset_name (str): e.g. "DIS-CO/MovieTection_Mini"
            results_base_folder (str): path to the folder that contains the MovieGuessTask model outputs.
            metrics_output_directory (str): path to the folder where the metrics will be saved.
        """
        self.method = method
        self.models = models
        self.dataset_name = dataset_name
        self.results_base_folder = results_base_folder
        self.metrics_output_directory = metrics_output_directory

        # Create results folder if needed.
        if not os.path.exists(self.metrics_output_directory):
            os.makedirs(self.metrics_output_directory)

    def run(self):
        """
        Performs the metrics computation:

          1) get_movie_answer_dictionaries -> obtains lists of clean and list of suspect movies.
          2) process_movie_data -> obtains main/neutral accuracy for each model.
          3) compute_auc -> calculates AUC for main/neutral.

        Finally, writes three JSON files:
          - <metrics_output_directory>/Results_Suspect_Data_Accuracy_<method>.json
          - <metrics_output_directory>/Results_Clean_Data_Accuracy_<method>.json
          - <metrics_output_directory>/Results_AUC_<method>.json
        """
        # 1) Get dictionaries for clean and suspect movies
        movie_list_clean, movie_list_suspect = get_movie_answer_dictionaries(
            dataset_name=self.dataset_name,
            results_path=self.results_base_folder
        )

        # 2) Process suspect data
        suspect_data = process_movie_data(
            movie_list=movie_list_suspect,
            models=self.models,
            path=self.results_base_folder,
            method=self.method
        )

        # 3) Process clean data
        clean_data = process_movie_data(
            movie_list=movie_list_clean,
            models=self.models,
            path=self.results_base_folder,
            method=self.method
        )

        # Write JSON outputs
        suspect_json_path = os.path.join(
            self.metrics_output_directory,
            f"Results_Suspect_Data_Accuracy_{self.method}.json"
        )
        clean_json_path = os.path.join(
            self.metrics_output_directory,
            f"Results_Clean_Data_Accuracy_{self.method}.json"
        )

        with open(suspect_json_path, "w") as json_file:
            json_file.write(json.dumps(suspect_data, indent=2))

        with open(clean_json_path, "w") as json_file:
            json_file.write(json.dumps(clean_data, indent=2))

        # 4) Compute & save AUC results
        auc_results = compute_auc(clean_data, suspect_data)
        auc_json_path = os.path.join(
            self.metrics_output_directory,
            f"Results_AUC_{self.method}.json"
        )
        with open(auc_json_path, "w") as json_file:
            json_file.write(json.dumps(auc_results, indent=2))

        print("\nMetric calculations complete.")
