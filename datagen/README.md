## Usage

The script needs access to an open ai api key, which will look for in an .env file.

Run the following command on console. 
```
python gpt_generator.py --input_path csv_file_name.csv --output_path csv_file_name_out.csv --prompt_path prompts/cloze_lab_prompt.txt
```

Where the input path is the name of the file with the missing column, the output path is the name of the file to print results to, and lastly the prompt path is the predefined prompt template to query the gpt. 
