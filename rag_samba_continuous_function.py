import fitz
import numpy as np
import openai
import requests
import torch
from transformers import (AutoTokenizer, DPRContextEncoder,
                          DPRContextEncoderTokenizer, DPRQuestionEncoder,
                          DPRQuestionEncoderTokenizer, LlamaTokenizer,
                          MistralForCausalLM)

client = openai.OpenAI(
    api_key="100cfa62-287e-4983-8986-010da6320a53",
    base_url="https://api.sambanova.ai/v1",
)


# url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
# response = requests.get(url)
# text = response.text

# with open('output.txt', 'r', encoding='utf-8') as file:
#     text = file.read()

# # Split by the phrase "Notebook PC E-Manual", assuming it appears once per page
# chunks = text.split('Notebook PC E-Manual')


def extract_section(pdf, start_page, end_page, title):
    section_text = ""
    for page_num in range(
        start_page - 1, end_page
    ):  # Page numbers are 0-indexed in PyMuPDF
        page = pdf.load_page(page_num)
        section_text += page.get_text()

    return section_text


def get_chunks(pdf_path):
    pdf_document = fitz.open(pdf_path)
    toc = pdf_document.get_toc()
    if toc:
        sections_list = []
        for i in range(len(toc)):
            title = toc[i][1]
            start_page = toc[i][2]
            end_page = (
                toc[i + 1][2] - 1 if i + 1 < len(toc) else pdf_document.page_count
            )  # Get end page based on next TOC entry

            # Extract text for the section
            section_text = extract_section(pdf_document, start_page, end_page, title)

            if section_text:
                sections_list.append(section_text)
    else:
        sections_list = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)  # Load the page
            page_text = page.get_text("text")  # Extract text from the page
            sections_list.append(page_text)

    chunks = sections_list
    return chunks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encode the context chunks
def encode_chunks(chunks):
    context_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    ).to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    )

    # Tokenize and encode context chunks
    context_encodings = []
    for chunk in chunks:
        inputs = context_tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            context_encoding = context_encoder(**inputs).pooler_output
        context_encodings.append(context_encoding.cpu().numpy())
    return np.vstack(context_encodings)


# Define the function to retrieve context
def retrieve_context(question, chunks, context_encodings):
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-multiset-base"
    ).to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-multiset-base"
    )

    # Tokenize and encode the question
    inputs = question_tokenizer(
        question, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        question_encoding = question_encoder(**inputs).pooler_output.cpu().numpy()

    # Compute similarities between the question encoding and all context encodings
    similarities = np.dot(context_encodings, question_encoding.T).flatten()

    # Find the most similar context
    most_similar_idx = similarities.argmax()
    return chunks[most_similar_idx]


# Function to generate a response based on a custom user message and conversation history
def generate_response(user_message, conversation_history, pdf_path="",chunks=None,context_encodings=None):
    user_dict = {"role": "user", "content": user_message}
    conversation_history.append(user_dict)
    # chunks = get_chunks(pdf_path)
    # context_encodings = encode_chunks(chunks)
    retrieved_context = retrieve_context(user_message, chunks, context_encodings)
    # print("_"*50)
    # print(retrieved_context)
    # print("_"*50)
    if retrieved_context:
        system_dict = {"role": "system", "content": retrieved_context}
        conversation_history.append(system_dict)

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


# pdf_path=["E:\ship\meheraab 3\manuals\preprocess\hp spectre x360 14 UM.pdf","E:\ship\meheraab 3\manuals\preprocess\hp spectre x360 14 SM.pdf"]

# total_chunks=[]


# for i in pdf_path:
#     chunks=get_chunks(i)
#     total_chunks.extend(chunks)

# print(type(total_chunks[0]))

# f = open("total_chunks.txt", "w")
# for i in total_chunks:
#     f.write("@"*120+"\n")
#     try:
#         f.write(i)
#     except UnicodeEncodeError:
#         f.write(i.encode('ascii', 'ignore').decode('ascii'))
# f.close()

# context_encodings=encode_chunks(total_chunks)
# while True:
#     user_message=input("Enter your message: ")
#     conversation_history=[]
#     response, conversation_history = generate_response(user_message, conversation_history, pdf_path,total_chunks,context_encodings)
#     print("*"*50)
#     print(response)
#     print("*"*50)






