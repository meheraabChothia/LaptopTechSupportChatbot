from fuzzywuzzy import process
import pandas as pd

def find_make_and_model(user_input, make=None):

    words=user_input.split()
    words = [word.strip() for word in words]
    print(words)

    df = pd.read_csv('global_models.csv')
    closest_make = closest_model =  pdf_path = ""
    make_score = model_score = 0

    for i in words:
        
        if i.lower() in df['company'].unique():
            make = i.lower()
            make_score = 100
            break
    
    # If make is provided, use it directly
    if make:
        closest_make = make
        make_score = 100
    else:
        # Extract unique companies (makes) from the DataFrame
        unique_makes = df['company'].unique()
        # Apply fuzzy matching to find the closest company/make
        closest_make, make_score = process.extractOne(user_input, unique_makes)
    
    # If we have a good make match
    if make_score > 80:
        # Filter the DataFrame to get models of the closest matching company
        company_models = df[df['company'] == closest_make]['model'].tolist()
        
        # Apply fuzzy matching to find the closest model from the selected company
        closest_model, model_score = process.extractOne(user_input, company_models)
        
        # Find the index of the row that matches both the closest make and model
        row_index = df.loc[(df['company'] == closest_make) & (df['model'] == closest_model)].index
        
        # Get the PDF path from the 'location' column at the found index
        if not row_index.empty:
            pdf_path = df.loc[row_index[0], 'location'].split(',')
    print(closest_make, closest_model, make_score, model_score, pdf_path)
    return closest_make, closest_model, make_score, model_score, pdf_path

# while True:
#     user_input = input("What is your model number?\n")
#     closest_make, closest_model, make_score, model_score, pdf_path = find_make_and_model(user_input)
#     print(closest_make, closest_model, make_score, model_score, pdf_path)
