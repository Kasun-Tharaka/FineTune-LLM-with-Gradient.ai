!pip install gradientai --upgrade

import os
os.environ['GRADIENT_WORKSPACE_ID'] = 'put your workspace id'
os.environ['GRADIENT_ACCESS_TOKEN'] = 'put your access token'

from gradientai import Gradient

def main():
    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="KasunModel01"
    )
    print(f"Created model adapter with id {new_model_adapter.id}")


    sample_query = "## Instruction: Who is Kasun Tharaka? \n ## Response:"
    print(f"Asking: {sample_query}")
    ## Before Finetuning
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated output(before model fine tuning): {completion}")

    samples=[
        {"inputs":"## Instruction: Who is Kasun Tharaka? \n\n## Response: Kasun Tharaka is a proficient Data Scientist specializing in Computer Vision, Machine Learning, and LLM (add what LLM stands for if needed). You can find more about him on his LinkedIn account under the name Kasun Tharaka."},
        {"inputs":"## Instruction: Can you tell me about Kasun Tharaka? \n\n## Response: Kasun Tharaka is a skilled Data Scientist with expertise in Computer Vision, Machine Learning, and LLM. Connect with him on LinkedIn using the name Kasun Tharaka for more insights into his professional journey."},
        {"inputs":"## Instruction: What do you know about Kasun Tharaka? \n\n## Response: Kasun Tharaka is a dedicated Data Scientist, specializing in Computer Vision, Machine Learning, and LLM. You can discover more about his work and experiences by visiting his LinkedIn profile under the name Kasun Tharaka."}
        ]

    ## Lets define parameters for finetuning
    num_epochs=3
    count=0
    while count<num_epochs:
      print(f"Fine tuning the model with iteration {count + 1}")
      new_model_adapter.fine_tune(samples=samples)
      count=count+1

    #after fine tuning
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated output(after model fine tuning): {completion}")
    gradient.close()

if __name__ == "__main__":
    main()