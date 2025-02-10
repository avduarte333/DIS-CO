"""
movie_guess_utils.py

This module contains the core functionality for the movie guessing task.
It encapsulates all helper functions, model and dataset initialization, and the 
logic for processing each movie (scenes, shots, etc.) into a class called MovieGuessTask.
"""

import base64
import os
import re
import json
import pandas as pd
from io import BytesIO
from PIL import Image  # Requires Pillow: pip install pillow
from tqdm import tqdm
from datasets import load_dataset  # Requires: pip install datasets


# Define a sample response schema (adjust this model as needed)
from pydantic import BaseModel
class MovieGuess(BaseModel):
    movie_name: str


class MovieGuessTask:
    def __init__(self, model_name, movie_option, frame_type, input_mode, clean_llm_output, results_base_folder, api_key, hf_auth_token):
        """
        Initializes the MovieGuessTask with configuration options.
        
        Args:
            model_name (str): The model name to use (e.g., "gpt-4o-2024-08-06").
            movie_option (str): Either "full" to process all movies, or the name of a single movie.
            frame_type (str): The frame type (e.g., "main" or "neutral").
            input_mode (str): "single_image" or "single_caption".
            clean_llm_output (bool): Whether to use a secondary cleaning API call.
            results_base_folder (str): Base folder to save the results.
        """
        self.model_name = model_name
        self.movie_option = movie_option
        self.frame_type = frame_type
        self.input_mode = input_mode
        self.clean_llm_output = clean_llm_output
        self.results_base_folder = results_base_folder
        self.api_key = api_key
        self.hf_auth_token = hf_auth_token

        # Create results folder if needed.
        if not os.path.exists(self.results_base_folder):
            os.makedirs(self.results_base_folder)

        # Initialize the clean client if cleaning is enabled.
        if self.clean_llm_output:
            if "gemini" in self.model_name.lower():
                from google import genai
                self.clean_client = genai.Client(api_key=self.api_key)

                # Create a config dictionary that includes both safety settings and JSON generation settings.
                self.clean_output_gemini_config = {
                    "safety_settings": [
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},],
                    "response_mime_type": "application/json",
                    "response_schema": MovieGuess,
                    "temperature" : 0.0}
            else:
                from openai import OpenAI
                self.clean_client = OpenAI(api_key = self.api_key)

        # Initialize the model client (or model/processor) based on the model_name.
        self.initialize_model_client()

        # Load the dataset.
        self.dataset = load_dataset("DIS-CO/MovieTection_Mini", split="train")

    def initialize_model_client(self):
        """
        Initializes model clients based on the provided model_name.
        Sets attributes such as self.client, self.model, self.processor, and self.model_type.
        """
        if "gemini" in self.model_name.lower():
            from google import genai

            # Create a config dictionary that includes both safety settings and JSON generation settings.
            self.gemini_config = {
                # Safety settings are provided as a list of dictionaries.
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                ],
                # JSON-generation settings:
                "response_mime_type": "application/json",
                # Use a list type hint with your Pydantic model to define the expected schema.
                "response_schema": MovieGuess,
                "temperature" : 0.0,
            }

            # Instantiate your client (ensure your API key is correctly set)
            self.client = genai.Client(api_key= self.api_key)
            self.model_type = "gemini"

        elif "gpt" in self.model_name.lower():
            from openai import OpenAI
            self.client = OpenAI(api_key = self.api_key)
            self.model_type = "gpt"

        elif "qwen" in self.model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            from qwen_vl_utils import process_vision_info  # Assumes you have this utility.

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                f"Qwen/{self.model_name}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir="/media/andrestorage"
            )
            self.processor = AutoProcessor.from_pretrained(f"Qwen/{self.model_name}")
            self.model_type = "qwen"
            self.process_vision_info = process_vision_info

        elif "llama" in self.model_name.lower():
            from huggingface_hub import login
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            import torch

            login(token=self.hf_auth_token)
            print("Successfully logged in!")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                f"meta-llama/{self.model_name}",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/media/andrestorage"
            )
            self.processor = AutoProcessor.from_pretrained(f"meta-llama/{self.model_name}")
            self.model_type = "llama"
        else:
            raise ValueError(f"Unsupported model type for model_name: {self.model_name}")



    def encode_image(self, image_input):
        """
        Encodes an image (provided as a file path, a PIL.Image, or a dictionary with 'bytes') to a base64 string.
        Llama 3.2, for some reason, does not seem to work with base64 encoded images, therefore, we return the PIL image directly.
        """
        # If the input is a file path (string), open the file directly.
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file {image_input} not found.")
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # If the input is a dictionary that contains image bytes (from the DataFrame conversion).
        elif isinstance(image_input, dict) and "bytes" in image_input:
            # Create a PIL Image from the raw bytes.
            image = Image.open(BytesIO(image_input["bytes"]))

            if self.model_type == "llama" or self.model_type == "gemini":
                return image
            else:
                # Convert to RGB if necessary.
                if image.mode != "RGB":
                    image = image.convert("RGB")
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # If the input is already a PIL image or a PIL-like object.
        elif hasattr(image_input, "convert") and hasattr(image_input, "mode"):
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")
            buffered = BytesIO()
            image_input.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        else:
            raise TypeError("Unsupported image input type. Expected file path, PIL.Image.Image, or dictionary with 'bytes'.")


    def clean_all_cells(self, df):
        """
        Cleans cells in the DataFrame by extracting text inside quotes and removing trailing periods.
        """
        def clean_cell(cell_value):
            if isinstance(cell_value, str):
                match = re.search(r'\"(.+?)\"', cell_value)
                if match:
                    result = match.group(1)
                    result = re.sub(r'\.$', '', result)
                    return result
            return cell_value

        return df.apply(lambda col: col.map(clean_cell))

    def clean_output(self, llm_output):
        """
        Cleans the raw LLM output using the dedicated cleaning API (an OpenAI GPT model).
        """
        cleaning_prompt = (
            "The following string is an output of a Language model that was tasked to identify a movie based on some images. "
            "Your task is to look at the entire output string and extract only the movie name present.\n"
            f"LLM Output: {llm_output}"
        )

        if self.model_type == "gemini":

            # Pass both your image and text content in the contents list.
            response = self.clean_client.models.generate_content(
                model="gemini-1.5-flash",
                contents= [cleaning_prompt],
                config=self.clean_output_gemini_config)
            try:
                output = json.loads(response.text)
            except Exception as e:
                raise Exception(f"Error parsing clean output: {e}")
            
        else:
            completion = self.clean_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                max_tokens=50,
                messages=[{"role": "user", "content": cleaning_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "movie_guess_task",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "movie_name": {
                                    "type": "string",
                                    "description": "Output only the movie name."
                                },
                            },
                            "required": ["movie_name"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            try:
                output = json.loads(completion.choices[0].message.content)
            except Exception as e:
                raise Exception(f"Error parsing clean output: {e}")


        return output['movie_name']

    def create_movie_guess_task(self, image_input=None, caption_input=None):
        """
        Creates the movie guessing task prompt, sends it to the model, and returns the guessed movie name.
        """
        image_prompt = (
            "The following image is a frame from a scene of a movie.\n"
            "Can you name the movie?\nYour guess must be only the movie name."
        )
        caption_prompt = (
            "The following caption describes a scene from a movie.\n"
            "Can you name the movie?\nYour guess must be only the movie name.\n\n"
            f"Caption: {caption_input}"
        )

        if self.model_type == "gemini":
            if image_input is not None:
                model_input = [image_prompt, self.encode_image(image_input)]
            elif caption_input is not None:
                model_input = caption_prompt
            else:
                raise ValueError("Either 'image_input' or 'caption_input' must be provided.")

            # Pass both your image and text content in the contents list.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents= model_input,
                config=self.gemini_config,
            )

            try:
                output = json.loads(response.text)
                guessed_movie = output["movie_name"]
            except Exception as e:
                raise Exception(f"Error parsing clean output: {e}")


        elif self.model_type == "gpt":
            if image_input is not None:
                try:
                    encoded_image = self.encode_image(image_input)
                except Exception as e:
                    raise Exception(f"Error encoding image: {e}")
                content = [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            elif caption_input is not None:
                content = [{"type": "text", "text": caption_prompt}]
            else:
                raise ValueError("Either 'image_input' or 'caption_input' must be provided.")

            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=50,
                messages=[{"role": "user", "content": content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "movie_guess_task",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "movie_name": {
                                    "type": "string",
                                    "description": "Make your guess for the movie in the image. Your guess must be only the movie name."
                                },
                            },
                            "required": ["movie_name"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            try:
                output = json.loads(completion.choices[0].message.content)
            except Exception as e:
                raise Exception(f"Error parsing GPT response: {e}")
            guessed_movie = output['movie_name']

        elif self.model_type == "qwen":
            if image_input is not None:
                try:
                    encoded_image = self.encode_image(image_input)
                except Exception as e:
                    raise Exception(f"Error encoding image for Qwen: {e}")
                content = [
                    {"type": "text", "text": image_prompt},
                    {"type": "image", "image": f"data:image/jpeg;base64,{encoded_image}"}
                ]
            elif caption_input is not None:
                content = [{"type": "text", "text": caption_prompt}]
            else:
                raise ValueError("Either 'image_input' or 'caption_input' must be provided.")

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p = None, top_k = None)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            guessed_movie = output_text[0]

        elif self.model_type == "llama":
            if image_input is not None:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": image_prompt}
                    ]}
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=text, images=self.encode_image(image_input), return_tensors="pt").to(self.model.device)
            elif caption_input is not None:
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": caption_prompt}
                    ]}
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=text, images=None, return_tensors="pt").to(self.model.device)
            else:
                raise ValueError("Either 'image_input' or 'caption_input' must be provided.")



            output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False , temperature = None, top_p = None)
            guessed_movie = self.processor.decode(output[0]).strip()

        else:
            raise ValueError(f"Unsupported model type for model_name: {self.model_name}")

        if self.clean_llm_output:
            guessed_movie = self.clean_output(guessed_movie)

        return guessed_movie

    def process_movie(self, movie_name_to_process):
        """
        Processes a single movie: filters the dataset for the given movie and frame type,
        iterates over scenes and shots, obtains the guessed movie name for each shot,
        groups the results, and finally saves the output as an Excel file.
        """
        # Convert the Hugging Face dataset to a Pandas DataFrame -> Filtering gets way faster than just using HuggingFace filtering.
        df_data = self.dataset.to_pandas()

        # Vectorized filtering using Pandas
        mask = (df_data['Movie'].str.lower() == movie_name_to_process.lower()) & (df_data['Frame_Type'].str.lower() == self.frame_type.lower())
        df_filtered = df_data[mask].copy()

        if df_filtered.empty:
            print(f"No data found for movie '{movie_name_to_process}' with frame type '{self.frame_type}'.")
            return

        # Sort the filtered DataFrame
        df_filtered.sort_values(by=['Scene_Number', 'Shot_Number'], inplace=True)

        results = []
        grouped = list(df_filtered.groupby('Scene_Number'))
        for scene_num, group in tqdm(grouped, total=len(grouped), desc="Processing Scenes"):
            scene_dict = {'Scene': scene_num}
            group_sorted = group.sort_values(by='Shot_Number')
            for idx, row in enumerate(group_sorted.itertuples(index=False)):
                if self.input_mode == "single_image":
                    image_input = row.Image_File
                    if isinstance(image_input, str) and not os.path.exists(image_input):
                        print(f"Image file {image_input} not found; skipping shot {idx+1} in scene {scene_num}.")
                        continue
                    try:
                        guessed_movie = self.create_movie_guess_task(image_input=image_input)
                    except Exception as e:
                        print(f"Error processing image in shot {idx+1} of scene {scene_num}: {e}")
                        continue
                    scene_dict[f"Image {idx+1}"] = guessed_movie
                elif self.input_mode == "single_caption":
                    caption = getattr(row, 'Caption', None)
                    if not caption or not isinstance(caption, str):
                        print(f"No valid caption found in shot {idx+1} of scene {scene_num}; skipping.")
                        continue
                    try:
                        guessed_movie = self.create_movie_guess_task(caption_input=caption)
                    except Exception as e:
                        print(f"Error processing caption in shot {idx+1} of scene {scene_num}: {e}")
                        continue
                    scene_dict[f"Caption {idx+1}"] = guessed_movie
                else:
                    raise ValueError("input_mode must be either 'single_image' or 'single_caption'")
            results.append(scene_dict)

        # Clean and save results
        results_df = pd.DataFrame(results)
        cleaned_df = self.clean_all_cells(results_df)

        movie_folder = os.path.join(self.results_base_folder, movie_name_to_process)
        if not os.path.exists(movie_folder):
            os.makedirs(movie_folder)

        output_filename = os.path.join(
            movie_folder, f"{movie_name_to_process}_results_{self.frame_type}_{self.model_name}_{self.input_mode}.xlsx"
        )
        cleaned_df.to_excel(output_filename, index=False)
        print(f"Results saved to {output_filename}")


    def run(self):
        """
        Main function to process movies. If movie_option is "full", every unique movie
        in the dataset is processed; otherwise only the specified movie is processed.
        """
        if self.movie_option.lower() == "full":
            unique_movies = pd.DataFrame(self.dataset)["Movie"].unique()
            total = len(unique_movies)
            for count, movie in enumerate(unique_movies, start=1):
                print(f"Starting processing for movie: {movie} ({count}/{total})")
                self.process_movie(movie)
                print("\n")
        else:
            self.process_movie(self.movie_option)