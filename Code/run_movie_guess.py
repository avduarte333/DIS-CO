"""
run_movie_guess.py

This is the main entry point for the movie guessing task. The purpose of this
script is to configure and run the task by setting a few key parameters.
All the underlying logic for processing the movies is encapsulated in the
movie_guess_utils.py module.

User-Defined Parameters:
-------------------------
model_name:
    - Controls which model will be used for the movie guessing task.
    - Options include:
        * "gpt-4o-2024-08-06" : OpenAI's GPT-4 model optimized for movie guessing.
        * "gemini-1.5-flash"  : Google's Gemini model.
        * "Qwen2-VL-7B-Instruct": Qwen2 for visual-language tasks.
        * "Llama-3.2-11B-Vision-Instruct": LLaMA 3.2 for vision-based tasks.

movie_option:
    - Determines which movie(s) will be processed.
    - Options:
        * "full"      : Process every movie in the dataset.
        * "<movie>"   : Process only the specified movie (e.g., "Frozen").

frame_type:
    - Specifies which type of frame from the movie should be processed.
    - Options include:
        * "main"      : Process the primary frames.
        * "neutral"   : Process neutral frames.

input_mode:
    - Selects the form of input for the task.
    - Options:
        * "single_image"   : Use an image file as input.
        * "single_caption" : Use a text caption as input.

clean_llm_output:
    - A boolean flag that controls whether the output of the language model
      should be cleaned/refined by an additional API call.
    - Options:
        * True  : Perform additional cleaning of the LLM output.
        * False : Use the raw output from the primary model.

results_base_folder:
    - Specifies the directory where the results (e.g., Excel files) will be saved.
    - Example: "./results" creates (or uses) a folder named "results" in the project directory.

api_key:
    - Your API key for closed-source models or when using the cleaning step.
    - Expect to provide either your OpenAI API key or Gemini API key here.

hf_auth_token:
    - Your Hugging Face authentication token.
    - Required when using LLaMA 3.2 models.

After configuring these parameters as needed, simply run this script to
execute the movie guessing task.
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
    results_base_folder="./results",
    api_key="YOUR_API_KEY",
    hf_auth_token="HF_ACCESS_TOKEN",
)

# Execute the movie guessing task.
task.run()
