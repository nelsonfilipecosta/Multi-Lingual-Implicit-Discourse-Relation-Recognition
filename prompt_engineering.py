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


def get_lower_level_predictions(predictions_l3):
    synchronous = predictions_l3[0]
    asynchronous = predictions_l3[1]+predictions_l3[2]
    cause = predictions_l3[3] + predictions_l3[4]
    condtion = predictions_l3[5] + predictions_l3[6]
    neg_condition = predictions_l3[7] + predictions_l3[8]
    purpose = predictions_l3[9] + predictions_l3[10]
    concession = predictions_l3[11] + predictions_l3[12]
    contrast = predictions_l3[13]
    similarity = predictions_l3[14]
    conjunction = predictions_l3[15]
    disjunction = predictions_l3[16]
    equivalence = predictions_l3[17]
    exception = predictions_l3[18] + predictions_l3[19]
    instantiation = predictions_l3[20] + predictions_l3[21]
    level_of_detail = predictions_l3[22] + predictions_l3[23]
    manner = predictions_l3[24] + predictions_l3[25]
    substitution = predictions_l3[26] + predictions_l3[27]

    predictions_l2 = [synchronous, asynchronous, cause, condtion, neg_condition, purpose,
                      concession, contrast, similarity, conjunction, disjunction, equivalence,
                      exception, instantiation, level_of_detail, manner, substitution]

    temporal = synchronous + asynchronous
    contingency = cause + condtion + neg_condition + purpose
    comparison = concession + contrast + similarity
    expansion = conjunction + disjunction + equivalence + exception + instantiation + level_of_detail + manner + substitution

    predictions_l1 = [temporal, contingency, comparison, expansion]

    return predictions_l1, predictions_l2


def get_js_distance(response, labels):
    shifted_labels = [x + 0.001 for x in labels]
    shifted_predictions = [x + 0.001 for x in response]

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
    labels_l1 = df.iloc[:,9:13].values.tolist()
    labels_l2 = df.iloc[:,14:31].values.tolist()
    labels_l3 = df.iloc[:,32:60].values.tolist()

    if LANG == "en":
        from Prompts.examples import examples_en as examples

    js_distance_l1 = 0
    js_distance_l2 = 0
    js_distance_l3 = 0

    for i in range(3):
        main_prompt = open(prompt_path).read()
        main_response=  open(response_path).read()
        question = create_question(arg_1[i], arg_2[i])

        messages = create_prompt(main_prompt, main_response, EXAMPLES_NUMBER)
        messages.append({"role": "user", "content": question})

        response = prompt_llm("gpt-4o-mini", messages)

        predictions_l3 = ast.literal_eval(response)
        predictions_l1, predictions_l2 = get_lower_level_predictions(predictions_l3)

        js_distance_l1 += get_js_distance(predictions_l1, labels_l1[i])
        js_distance_l2 += get_js_distance(predictions_l2, labels_l2[i])
        js_distance_l3 += get_js_distance(predictions_l3, labels_l3[i])
    
    js_distance_l1 /= 3
    js_distance_l2 /= 3
    js_distance_l3 /= 3

    print("Level-1: ", js_distance_l1)
    print("Level-2: ", js_distance_l2)
    print("Level-3: ", js_distance_l3)