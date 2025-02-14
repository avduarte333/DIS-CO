"""
The purpose of this script is to obtain the metrics after having generated the movie predictions with the MovieGuessTask.
The calculated metrics are:
    - Movie-Level Accuracy: Accuracy for individual movies, reported separately for Suspect and Clean movies.
    - Overall Accuracy: The average accuracy across all movies, reported separately for the Suspect and Clean splits.
    - Dataset AUC: The Area Under the Curve (AUC) based on the movie-level accuracies for Suspect and Clean movies.
 


User-Defined Parameters:
-------------------------
method:
    - Selects the form of input for the task.
    - Options:
        -'disco': Uses the single image files (_single_image.xlsx).

        -'captions': Uses the single caption files (_single_caption.xlsx).

        -'disco_floor': Loads both the single_image and the single_caption files.
            Aligns them (ensuring they have the same shape) and replaces matching cells with the placeholder 'XX-REMOVED-XX' in the image DataFrame before calculating accuracy. 
            Provides a more strict classification of possible suspect movies as it only considers frames that were not correctly classified by the caption data.

models:
    - A list of the models for which we want the metrics.
    - Example: ['gpt-4o-2024-08-06', 'gemini-1.5-pro', 'Qwen2-VL-72B-Instruct', ...]


dataset_name:
    - The dataset needed to obtain the ['Movie', 'Answer'] pairs into a dictionary format used for evaluation. If using movies not present in MovieTection or MovieTecion_Mini, the "get_movie_answer_dictionaries" function from metrics_utils should be adjusted accordingly.
    - Current options include:
        * "DIS-CO/MovieTection"       : The full movie dataset.
        * "DIS-CO/MovieTection_Mini"  : A smaller subset.

results_base_folder:
    - The directory where the results of the MovieGuessTask were saved and where the metrics will be saved.
    - Example: "./Results" creates (or uses) a folder named "results" in the project directory.

metrics_output_directory:
    - The directory where the metrics will be saved.
    - Example: "./Metrics" creates (or uses) a folder named "Metrics" in the project directory.
-------------------------


If you wish to replicate the DIS-CO results presented in Tables 2, 11, and 12, use the following parameters. Please note that minor deviations may occur, as this script does not apply the sampling with replacement for the sake of clearer movie-level results interpretation.

metrics = Metrics(
    method="disco",
    models=['gpt-4o-2024-08-06', 'gemini-1.5-pro', 'Llama-3.2-90B-Vision-Instruct', 'Qwen2-VL-72B-Instruct'],
    dataset_name="DIS-CO/MovieTection",
    results_base_folder="./Replicate_Results",
    metrics_output_directory="./Metrics"
)
metrics.run() 
"""

from metrics_utils import Metrics


metrics = Metrics(
    method="disco",
    models=['gpt-4o-2024-08-06', 'gemini-1.5-pro', 'Llama-3.2-90B-Vision-Instruct', 'Qwen2-VL-72B-Instruct'],
    dataset_name="DIS-CO/MovieTection",
    results_base_folder="./Replicate_Results",
    metrics_output_directory="./Metrics"
)
metrics.run()
