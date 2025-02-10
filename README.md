# DIS-CO 🪩

This is the official repository for the paper DIS-CO: Discovering Copyrighted Content in VLMs Training Data by André V. Duarte, Xuandong Zhao, Arlindo L. Oliveira and Lei Li<br>

DIS-CO is a method ...

![DIS-CO](DISCO-Teaser.png)



---
## DIS-CO Example
⚠ Important: When using Gemini, ChatGPT, LLaMA 3.2 or any other model that requires an API key / HF Login Credentials, don't forget to add your own keys in the xxxxxxx.py file<br>

```python
from movie_guess_utils import MovieGuessTask

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
```
---

### 📚 [MovieTection](https://huggingface.co/datasets/DIS-CO/MovieTection)
MovieTection consists of movie frames manually extracted from Project Gutenberg and subsequently segmented using LumberChunker.<br>
It features: **100 xxx** and **30 xxx** per Movie.<br>

The dataset is organized into the following columns:
- `Movie`: \<...\>
- `Frame_Type`: \<...\>
- `Scene_Number`: \<...\>
- `Shot_Number`: \<...\>
- `Image_File`: \<...\>
- `Caption`: \<...\>
- `Label`: \<...\>
- `Answer`: \<...\>



---
### 📖 MovieTection Alternative (Used for Baseline Methods)
We also release the same corpus on a subset format containing 4 Movies only.



---
### 🤝 Compatibility
DIS-CO is compatible with any VLM with strong reasoning capabilities.<br>
- In our code, we provide an implementation for Gemini and ChatGPT, but in fact, models like LLaMA-3, Mixtral 8x7B, or Command+R can also be used.<br>


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
