# DIS-CO 🪩

This is the official repository for the paper DIS-CO: Discovering Copyrighted Content in VLMs Training Data by André V. Duarte, Xuandong Zhao, Arlindo L. Oliveira and Lei Li.<br>
Explore the project here! [DIS-CO Blog](https://avduarte333.github.io/projects/disco/)

### Overview
DIS-CO is a method designed to infer whether copyrighted content was included in the training data of vision-language models (VLMs). By querying models with frames from targeted copyrighted media and analyzing their free-form responses, DIS-CO can provide strong evidence of memorization while remaining applicable to both white-box and black-box models.

<p align="center">
  <img src="./Images/DISCO-Teaser.png" width="60%">
</p>



---

## 🚀 Instalation
To set up **DIS-CO**, follow these steps:

```bash
# Clone the repository
git clone https://github.com/avduarte333/DIS-CO
cd DIS-CO

# Create a conda environment
conda create -n disco_env python=3.10 pip -y
conda activate disco_env
```
---

## 🔹 Installing Dependencies

You can install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

- **To install dependencies for all supported models:**  
  Run the command as is.

- **To install dependencies for specific models only:**  
  Open the `requirements.txt` file and comment out the sections corresponding to the models you don’t need. Each model's dependencies are clearly labeled with comments for easy identification.


---

## 🗄️ [MovieTection](https://huggingface.co/datasets/DIS-CO/MovieTection)
The MovieTection dataset is designed for image/caption-based question-answering, where models predict the movie title given a frame or its corresponding textual description.

### Dataset Structure 🚧
The dataset consists of **14,000 frames** extracted from **100 movies**, categorized into:
- **Suspect movies:** Released *before* September 2023, **potentially** included **in training data**.
- **Clean movies:** Released *after* September 2023, **outside** the models' **training data**.

Each movie contains **140 frames**, classified into:
- **Main Frames**: Featuring key characters from the plot.
- **Neutral Frames**: Backgrounds, objects, or minor characters.


<p align="center">
  <img src="./Images/image_types_example.png" alt="Image 1" width="85%">
</p>



The dataset is organized into the following columns:

- `Movie` - The title of the movie from which the frame was extracted.

- `Frame_Type` - Categorization of the frame as either *Main* or *Neutral*.

- `Scene_Number` - Identifier for the scene within the movie. Scenes group related frames together based on their narrative context.

- `Shot_Number` - The identifier for the specific shot within a scene.

- `Image_File` - The image file of the frame.

- `Caption` - Detailed textual description of the frame, generated using Qwen2-VL 7B.

- `Label` - Binary value indicating whether the movie is categorized as clean (0) or suspect (1).

- `Answer` - List of acceptable movie title variations that should be considered correct when evaluating the model’s free-form responses.



---
## 📁 [MovieTection_Mini](https://huggingface.co/datasets/DIS-CO/MovieTection_Mini) - Dataset Alternative
This dataset is a compact subset of the full MovieTection dataset, containing only 4 movies instead of 100. It is designed for users who want to experiment with the benchmark without the need to download the entire dataset, making it a more lightweight alternative for testing.

---



## 🎬 Running DIS-CO

### Obtaining Movie Predictions
To run the **movie guessing task**, fill the attributes of `run_movie_guess.py` according to your needs:

```python
from movie_guess_utils import MovieGuessTask

task = MovieGuessTask(
    model_name = "gpt-4o-2024-08-06",  # The model name to use
    movie_option = "full",             # Either "full" to process all movies, or the name of a single movie.
    frame_type = "main",               # The frame type (e.g., "main" or "neutral").
    input_mode = "single_image",       # Expects "single_image" or "single_caption".
    clean_llm_output = False,          # Whether to use a secondary cleaning API call.
    results_base_folder = "./Results", # Base folder to save the results.
    api_key = "YOUR_API_KEY",          # API key for OpenAI or Gemini models.
    hf_auth_token = "HF_ACCESS_TOKEN", # Hugging Face authentication token - Required when using LLaMA 3.2 models.
    dataset = "DIS-CO/MovieTection"    # Dataset name (e.g., "DIS-CO/MovieTection" or "DIS-CO/MovieTection_Mini").
)

# Execute the movie guessing task.
task.run()
```

Once set up, launch the task with:

```bash
python Code/run_movie_guess.py
```

### Calculating Metrics
Once we have obtained predictions for different movies, the next step is to evaluate the model's performance.<br>
**Note**: If you only wish to replicate the DIS-CO results presented in Tables 2, 11, and 12, you can use the parameters provided in the code example.

```python
from metrics_utils import Metrics

metrics = Metrics(
    method = "disco",                             # Methods: "captions", "disco", or "disco_floor"
    models = [                                    # List of model names
        'gpt-4o-2024-08-06', 
        'gemini-1.5-pro', 
        'Llama-3.2-90B-Vision-Instruct', 
        'Qwen2-VL-72B-Instruct'
    ],                                          
    dataset_name = "DIS-CO/MovieTection",         # Dataset name (e.g., "DIS-CO/MovieTection" or "DIS-CO/MovieTection_Mini").
    results_base_folder = "./Replicate_Results",  # Path to the folder that contains the MovieGuessTask model outputs.
    metrics_output_directory = "./Metrics"        # Path to the folder where the metrics will be saved.
)

metrics.run()

```

Once set up, launch the task with:

```bash
python Code/run_metrics.py
```

---
## 💬 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{duarte_disco,
      title={DIS-CO: Discovering Copyrighted Content in VLMs Training Data}, 
      author={André V. Duarte and Xuandong Zhao, Arlindo L. Oliveira and Lei Li},
      year={2025},
      eprint={xxxx.yyyyy},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/xxxx.yyyyy}, 
}
```
