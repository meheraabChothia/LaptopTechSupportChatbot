import os

import openai

client = openai.OpenAI(
    api_key="100cfa62-287e-4983-8986-010da6320a53",
    base_url="https://api.sambanova.ai/v1",
)


def generate_response(user_message: str, conversation_history: dict, pdf_path="",chunks=None,context_encodings=None) -> tuple[str, dict]:
    """
    Will send the user's message to the LLM and obtain the response.
    
    Args:
        user_message (str): A string that holds the user's message to the LLM.
        conversation_history (dict): A dictionary that holds the conversation history in the required format.
        pdf_path (str): A string that holds the path to the PDF file.
        
    Returns:
        tuple[str, dict]: A tuple that holds the response from the LLM and the updated conversation history.
    """
    user_dict = {"role": "user", "content": user_message}
    conversation_history.append(user_dict)
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=conversation_history,
        temperature=0.1,
        top_p=0.1,
    )
    response = response.choices[0].message.content
    assistant_dict = {"role": "assistant", "content": response}
    conversation_history.append(assistant_dict)
    return response, conversation_history


# system_prompt = [{"role":"system","content":"You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."}]
# conversation_history = system_prompt
# print("You can start the conversation now. Type 'exit' to stop.")
# while True:
#     # Get user input
#     print("________________________________________________________________________________________________________________________________________________________________________________________________")
#     custom_user_message = input("You: ")
#     print("________________________________________________________________________________________________________________________________________________________________________________________________")
#     # Check if the user wants to exit
#     if custom_user_message.lower() == 'exit':
#         print("Ending conversation.")
#         break

#     if custom_user_message.lower() == 'print':
#         print(conversation_history)
#         continue

#     if custom_user_message.lower() == 'clear':
#         conversation_history=system_prompt
#         print("History Cleared.")
#         continue

#     # Generate and print the model's response
#     response, conversation_history = generate_response(custom_user_message, conversation_history)
#     print("________________________________________________________________________________________________________________________________________________________________________________________________")
#     print(f"Assistant: {response}")
