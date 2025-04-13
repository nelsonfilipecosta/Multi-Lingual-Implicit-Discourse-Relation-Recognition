import os
import sys
import ast
import pandas as pd
import openai
from scipy.spatial.distance import jensenshannon


MODE = sys.argv[1]
if MODE not in ["validation", "test"]:
    print("Type a valid mode: validation or test.")
    exit()

LANG = sys.argv[2]
if LANG not in ["all", "en", "de", "fr", "cs"]:
    print("Type a valid language: all, en, de, fr or cs.")
    exit()

EXAMPLES_NUMBER = 5


openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


def create_prompt(main_prompt, main_response, examples_number=0):
    messages = [{"role": "user",      "content": main_prompt},
                {"role": "assistant", "content": main_response}]

    if examples_number > 0:
        for i in range(examples_number):
            messages.append({"role": "user",      "content": examples["question"][i]})
            messages.append({"role": "assistant", "content": examples["answer"][i]})

    return messages


def create_question(arg_1, arg_2):
    question = f"Sentence 1: {arg_1}\n"\
               f"Sentence 2: {arg_2}"

    return question


def prompt_llm(model, messages):
    response = client.chat.completions.create(model = model,
                                              messages = messages,
                                              temperature = 0,
                                              max_tokens = 1024)

    return response.choices[0].message.content


def get_js_distance(response, labels):
    shifted_labels = [x + 0.001 for x in labels]
    shifted_predictions = [x + 0.001 for x in response]

    print(shifted_labels)
    print(shifted_predictions)

    js_distance = jensenshannon(shifted_labels, shifted_predictions, base=2)

    return js_distance


if __name__ == "__main__":

    if MODE == "validation":
        df = pd.read_csv("Data/DiscoGeM-2.0/discogem_2_single_lang_" + LANG + "_validation.csv")
    elif MODE == "test":
        df = pd.read_csv("Data/DiscoGeM-2.0/discogem_2_single_lang_" + LANG + "_test.csv")

    prompt_path = "Prompts/main_prompt_" + LANG + ".txt"
    response_path = "Prompts/main_response_" + LANG + ".txt"

    arg_1 = df["arg1"].tolist()
    arg_2 = df["arg2"].tolist()
    labels = df.iloc[:,32:60].values.tolist()

    if LANG == "en":
        from Prompts.examples import examples_en as examples

    for i in range(3):
        main_prompt = open(prompt_path).read()
        main_response=  open(response_path).read()
        question = create_question(arg_1[i], arg_2[i])

        messages = create_prompt(main_prompt, main_response, EXAMPLES_NUMBER)
        messages.append({"role": "user", "content": question})

        response = prompt_llm("gpt-4o-mini", messages)
        predictions_l3 = ast.literal_eval(response)

        print(get_js_distance(predictions_l3, labels[i]))