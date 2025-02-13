"""
The purpose of this script is to configure and run the task by setting a few key parameters.
All the underlying logic for processing the movies is encapsulated in the movie_guess_utils.py module.


User-Defined Parameters:
-------------------------
model_name:
    - Controls which model will be used for the movie guessing task.
    - Current implementation allows for the following options:
        * "gpt-4o-2024-08-06" : OpenAI's GPT-4o model.
        * "gemini-1.5-flash"  : Google's Gemini model.
        * "Qwen2-VL-7B-Instruct": Alibaba's Qwen2-VL model.
        * "Llama-3.2-11B-Vision-Instruct": Meta's model.

movie_option:
    - Determines which movie(s) will be processed.
    - Options:
        * "full"      : Process every movie in the dataset.
        * "<movie>"   : Process only the specified movie (e.g., "Frozen").

frame_type:
    - Specifies which type of frame from the movie should be processed.
    - Options include:
        * "main"      : (i) Featuring key characters from the plot; (ii) Easily recognizable to viewers who saw the movie.
        * "neutral"   : (i) Backgrounds, objects, or minor characters; (ii) Frames not easily tied to the movie's narrative.

input_mode:
    - Selects the form of input for the task.
    - Options:
        * "single_image"   : Use an image file as input.
        * "single_caption" : Use a text caption as input.

clean_llm_output:
    - A flag that controls whether the output of the language model should be cleaned by an additional API call.
    - Our experience shows that LLaMA-3.2 model outputs can be noisy, and this step can help improve the quality of the results.
    - Options:
        * True  : Perform additional cleaning of the LLM output.
        * False : Use the raw output from the primary model.

results_base_folder:
    - Specifies the directory where the results (e.g., Excel files) will be saved.
    - Example: "./results" creates (or uses) a folder named "results" in the project directory.

api_key:
    - Specify an API key when using closed-source models or when using the cleaning step.
    - Expects either an OpenAI or Gemini API key. By default, OpenAI's API is used unless the inference model is Gemini.

hf_auth_token:
    - Hugging Face authentication token.
    - Required when using LLaMA 3.2 models.

dataset:
    - The dataset to use for the task.
    - Accepts DIS-CO/MovieTection or DIS-CO/MovieTection_Mini (a smaller version for users who want to experiment with the benchmark without downloading the full dataset).
"""

from movie_guess_utils import MovieGuessTask

# --------------------------------------------------------------------------
# User Configuration: Modify the parameters below to control the behavior
# of the movie guessing task.
# --------------------------------------------------------------------------
task = MovieGuessTask(
    model_name="gpt-4o-2024-08-06",
    movie_option="full",
    frame_type="main",
    input_mode="single_image",
    clean_llm_output=False,
    results_base_folder="./Results",
    api_key="YOUR_API_KEY",
    hf_auth_token="HF_ACCESS_TOKEN",
    dataset = "DIS-CO/MovieTection_Mini"
)

# Execute the movie guessing task.
task.run()
