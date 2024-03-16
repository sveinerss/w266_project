import click
import pandas as pd
from tqdm import tqdm
import os 
from dotenv import load_dotenv
from openai import OpenAI


def prompt_format(context_string: str, story: str, ending:str):
    # replace with current story's info
    context_string = context_string.replace('{Story}', story)
    context_string = context_string.replace('{CorrectE}', ending)
    return context_string

def saveContents(arr: list, output_path:str):
    pd.DataFrame(arr)[["storyid","storytitle","sentence1","sentence2","sentence3","sentence4","correctE","incorrectE"]].to_csv(output_path)

def request(
    prompt: str,
    model="gpt-3.5-turbo",
    max_tokens=60,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop='\n',
    presence_penalty=0.0,
    frequency_penalty=0.0
    ):
    '''
    Queries the gpt model to create relevant, contextual and factual information given the input. 
    '''
    # retry request (handles connection errors, timeouts, and overloaded API)
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(60)
    
    return response.choices[0].message.content


@click.command()
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--prompt_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=1)
@click.option('--top_p', default=1.0, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=60, type=int)
def main(
    input_path: str,
    output_path: str,
    prompt_path: str,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int
):
    
    # read examples for inference
    input_df = pd.read_csv(input_path)
    input_df["story"] = input_df.apply(lambda row: ' '.join(row[[f'sentence{x}' for x in range(1,5)]].astype(str)), axis=1)
    input_df.rename(columns={"sentence5":"correctE"}, inplace=True)
    

    # read prompt template
    with open(prompt_path) as f:
        prompt_text = f.read().strip('\n')

    generated_examples = []

    for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):


        try:
            context_string = prompt_format(
                prompt_text,
                story=row["story"],
                ending=row['correctE'])
                    
            prompt_to_pass = {'role':'user', 'content':context_string}

            incorrectEnding = request(
                prompt_to_pass,
                n=num_knowledge,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens)

            row['incorrectE'] = incorrectEnding
            generated_examples.append(row.to_dict())
            

        except KeyboardInterrupt:
            print("Interrupted! Saving contents...")
            saveContents(generated_examples, output_path)
                
    saveContents(generated_examples, output_path)

if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    main()