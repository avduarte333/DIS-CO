"""
run_movie_guess.py

This is the main script file.
A user who wants to run the movie guessing task on a new movie need only change
the following parameters: model_name, movie_option, frame_type, input_mode, and clean_llm_output.
Everything else is handled by movie_guess_utils.py.
"""

from movie_guess_utils import MovieGuessTask

# -------------------- User-Defined Parameters --------------------
model_name = "Qwen2-VL-7B-Instruct"      # Options: "gpt-4o-2024-08-06", "gemini-1.5-flash", "Qwen2-VL-7B-Instruct", "Llama-3.2-11B-Vision-Instruct"
movie_option = "Frozen"                  # Set to "full" to process all movies or specify a movie name (e.g., "Inception")
frame_type = "neutral"               # e.g., "main" or "neutral"
input_mode = "single_image"          # Options: "single_image" or "single_caption"
clean_llm_output = True              # Whether to use the secondary API call to clean the LLM output
results_base_folder = "./results"    # Base folder for saving the results
api_key = "YOUR_API_KEY"                  # Required for closed-source models or the clean_llm_output option. Expecting either OpenAI or Gemini Key.
hf_auth_token = "HF_ACCESS_TOKEN"      # Required for LLaMA 3.2 Model





# -------------------- Initialize and Run the Task --------------------
task = MovieGuessTask(
    model_name=model_name,
    movie_option=movie_option,
    frame_type=frame_type,
    input_mode=input_mode,
    clean_llm_output=clean_llm_output,
    results_base_folder=results_base_folder,
    api_key=api_key,
    hf_auth_token=hf_auth_token
)

task.run()