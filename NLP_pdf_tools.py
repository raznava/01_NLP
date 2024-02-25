##################################################
# NLP_pdf_tools.py
# Original author: MSc. Edgar Regulo Vega Carrasco
# Version: 1.0
# Last modification: 2023-11-02
# Description: 
#              1. feature_pdf_extractor
#              This script contains the functions to extract data from PDF files

##################################################
import os
import re
import fitz
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')

def feature_pdf_extractor(directory,path_output, name_file_output_csv,name_file_output_xlsx, feature_sort, number_articles, size_x, size_y):
    """
    
    """ 
    # Function to extract text from PDF file
    def extract_pdf_text(pdf_file):
        pdf_doc = fitz.open(pdf_file)
        text = ""
        for page in pdf_doc:
            text += page.get_text("text")
        pdf_doc.close()
        text = text.replace('\n', ' ')  # remove newline characters
        return text
    # Function to extract abstract from text from PDF file
    def extract_abstract(text):
        sentences = sent_tokenize(text)
        abstract = ""
        count = 0
        for sentence in sentences:
            if "abstract" in sentence.lower() or "a b s t r a c t" in sentence.lower():
                abstract = sentence + " "
                count += 1
            elif count > 0:
                abstract += sentence + " "
                count += 1
            if count == 10:
                break
        return abstract
    # Function to extract keywords from text from PDF file
    def extract_keywords(text):
        sentences = sent_tokenize(text)
        keywords = ""
        count = 0
        for sentence in sentences:
            if "keywords" in sentence.lower():
                words = word_tokenize(sentence)
                for word in words:
                    if count == 50:
                        break
                    if word.lower() not in ["keywords", ":"]:
                        keywords += word + " "
                        count += 1
        return keywords
    # Function to extract results from text from PDF file
    def extract_results(text):
        sentences = sent_tokenize(text)
        results = ""
        count = 0
        found = False
        for sentence in sentences:
            if "results" in sentence.lower() and not found:
                results = sentence + " "
                found = True
                count += 1
            elif found:
                results += sentence + " "
                count += 1
            if count == 100:
                break
        return results
    # Function to extract first two sentences from text from PDF file
    def extract_first_two_sentences(text):
        sentences = sent_tokenize(text)
        first_two = ""
        count = 0
        for sentence in sentences:
            first_two += sentence + " "
            count += 1
            if count == 2:
                break
        return first_two
    # Function to extract from sentence 3 to 14 from text from PDF file
    def extract_from_sentence_3_to_14(text):
        sentences = sent_tokenize(text)
        extracted = ""
        count = 0
        for sentence in sentences:
            if count >= 2 and count <= 13:
                extracted += sentence + " "
            count += 1
            if count == 14:
                break
        return extracted
    # Function to generate words dictionary
    def generate_words_dictionary(words):
        words_dictionary = {}
        for word in words:
            pattern = r'\b{}\b'.format(r'\s+'.join(word.split()))
            words_dictionary[word] = {
                'pattern': pattern,
                'context': []
            }
        return words_dictionary
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    # Create a list to store the data
    data = []
    column_names = ['file_name', 'title', 'abstract', 'results']
    # To sum the row values of the generated columns for each feature_word 
    def generate_feature_words(df):
        feature_words = []
        for row in range(0, df.shape[0]):
            feature_word = [df.iloc[row, 0]]
            feature_word_synonyms = df.iloc[row, 1:].dropna().tolist()
            feature_words.append((feature_word, feature_word_synonyms))
        return feature_words
    # Path to the feature file
    excel_features_file_path = "features.xlsx"
    # Read the Excel file into a DataFrame, considering the first row as data
    df_features = pd.read_excel(excel_features_file_path, header=None)
    # Generate feature words and their synonyms
    feature_words = generate_feature_words(df_features)
    # Extend column_names with dynamically generated column names for each feature_word and its synonyms
    for feature_word, feature_word_synonyms in feature_words:
        column_names.extend([f'{feature_word[0]}_present_{i+1}' for i in range(len(feature_word + feature_word_synonyms))])
        column_names.extend([f'{feature_word[0]}_context_{i+1}' for i in range(len(feature_word + feature_word_synonyms))])
    # Loop through the PDF files and apply the functions
    for pdf_file in pdf_files:
        words_to_detect_wrong_title = ['doi', 'No Job Name']
        pattern = '[0-9]{5,}'
        words_to_detect_wrong_title = [word for word in words_to_detect_wrong_title if not re.search(pattern, word)]
        file_path = os.path.join(directory, pdf_file)
        text = extract_pdf_text(file_path)
        pdf_doc = fitz.open(file_path)
        title = pdf_doc.metadata.get("title", "")
        abstract = extract_abstract(text)
        keywords = extract_keywords(text)
        results = extract_results(text)
        first_two = extract_first_two_sentences(text)
        from_sentence_3_to_14 = extract_from_sentence_3_to_14(text)
        title_improved = title if title and all(word not in title for word in words_to_detect_wrong_title) else first_two
        abstract_improved = abstract if abstract else from_sentence_3_to_14
        # Create a dictionary to store the data
        data_row = {
            'file_name': pdf_file,
            'title': title_improved,
            'abstract': abstract_improved,
            'results': results
        }
        # Loop through the feature_words and their synonyms
        for feature_word, feature_word_synonyms in feature_words:
            words_list = feature_word + feature_word_synonyms
            feature_word_and_synonyms = generate_words_dictionary(words_list)
            # Loop through the feature_word and its synonyms
            for i, (word, info) in enumerate(feature_word_and_synonyms.items()):
                pattern_word = re.compile(info['pattern'], re.IGNORECASE)
                matches = pattern_word.findall(text)
                count = len(matches)
                present = True if count > 0 else False
                if present:
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        if pattern_word.search(sentence):
                            info['context'].append(sentence)
            # Update the data_row with the generated columns for each feature_word and its synonyms
            data_row.update({f'{feature_word[0]}_present_{i+1}': len(info['context']) for i, (_, info) in enumerate(feature_word_and_synonyms.items())})
            data_row.update({f'{feature_word[0]}_context_{i+1}': info['context'] for i, (_, info) in enumerate(feature_word_and_synonyms.items())})
        # Append the data_row to the data list
        data.append(data_row)
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=column_names)
    # To sum the row values of the generated columns for each feature_word
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_present_')]
        df[f'{column_prefix}_present_sum'] = df[present_columns].sum(axis=1)
    # To sum the row values of the generated columns for each feature_word
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_context_')]
        df[f'{column_prefix}_context_sum'] = df[present_columns].sum(axis=1)
    # to delete duplicates in context_columns
    context_columns = [column_name for column_name in df.columns if column_name.endswith('_context_sum')]
    for column in context_columns:
        df[column] = df[column].apply(lambda x: list(set(x)) if isinstance(x, list) else [])
    # Sort according to the sum of present columns for each feature_word
    df = df.sort_values(by=[f'{feature_word[0]}_present_sum' for feature_word, _ in feature_words], ascending=False)
    # To sum the row values of the generated columns for each feature_word
    def generate_feature_unit_words(df):
        feature_unit_words = []
        for row in range(0, df.shape[0]):
            feature_unit_word = [df.iloc[row, 0]]
            feature_unit_word_synonyms = df.iloc[row, 1:].dropna().tolist()
            feature_unit_words.append((feature_unit_word, feature_unit_word_synonyms))
        return feature_unit_words
    # Path to the feature file
    excel_features_units_file_path = "features_units.xlsx"
    # Read the Excel file into a DataFrame, considering the first row as data
    df_features_units = pd.read_excel(excel_features_units_file_path, header=None)
    # Generate feature words and their synonyms
    feature_unit_words = generate_feature_unit_words(df_features_units)
    # Extend column_names with dynamically generated column names for each feature_unit_word and its synonyms
    def extract_number_1(lst, unit1):
        """Function to extract unique numbers integral or float followed by unit1, regardless of case"""
        if isinstance(lst, list):
            numbers = []
            for item in lst:
                if isinstance(item, str):
                    regex_list = [
                        # Pattern to extract a number followed by the unit: [number] unit
                        # match: '10 kg', '-3.14 m', '15.5Kg'. Extract: 10, -3.14, 15.5
                        fr'(-?\d+\.?\d*)\s*(?:{unit1})',
                        # Pattern to extract a number followed by ± number unit: [number]  ± number unit
                        # match: '20.5 ± 0.2 cm'. Extract: 20.5  
                        fr'(-?\d+(?:\.\d+)?)(?=\s*±\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to extract a number followed by (optional ,) and number  unit: [number] , number unit
                        # match: '5.2, and 8.9 cm' , '4.5 and 6 kg'. Extract: 5.2, 4.5
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] , number, and number unit 
                        # match: '1.1, 5.2, and 8.9 cm' , '1.2, 4.5 and 6 kg'. Extract: 1.1, 1.2
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, and number unit 
                        # match: '0.3, 1.1, 5.2, and 8.9 cm' , '0.5, 1.2, 4.5 and 6 kg'. Extract: 0.3, 0.5 
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, number,  and number unit 
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, number, number,  and number unit
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})'
                        # Pattern to ... [number] ,number, number, number, number, number,  and number unit
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})'
                    ]
                    numbers_set = set()
                    for regex in regex_list:
                        matches = re.findall(regex, item, re.IGNORECASE)
                        for match in matches:
                            if '.' in match:
                                numbers_set.add(float(match))
                            else:
                                numbers_set.add(int(match))
                    numbers_str = ', '.join(str(num) for num in numbers_set)
                    if numbers_str:
                        numbers.append(numbers_str)
                    else:
                        numbers.append("[]")
            return numbers
        else:
            return []
    # To put the numbers in the context columns
    for feature_unit_word, feature_unit_word_synonyms in feature_unit_words:
        prefix = feature_unit_word[0]
        prefix_context_sum = [column_name for column_name in df.columns if column_name.startswith(f'{prefix}_context_sum')]
        for unit1 in feature_unit_word_synonyms:
            df[f'{prefix}_context_number_{unit1}'] = df[prefix_context_sum].astype(str).apply(lambda x: extract_number_1(x.tolist(), unit1=unit1))
    # Collecting the numbers in the context_number columns
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_context_number')]
        df[f'{column_prefix}_number_collected'] = df[present_columns].sum(axis=1)
    # Function to remove duplicates from a comma-separated string of numbers
    def remove_duplicates(s):
        # Split the string into a list of numbers, strip whitespace just in case
        numbers = [num.strip() for num in s.split(',')]
        # Convert the list to a set to remove duplicates, then back to a list
        unique_numbers = list(set(numbers))
        # Join the unique numbers back into a comma-separated string
        return ','.join(unique_numbers)
    # to delete "[]" in _number_collected columns:
    number_collected_columns = [column_name for column_name in df.columns if column_name.endswith('_number_collected')]
    for column in number_collected_columns:
        df[column] = df[column].apply(lambda x: remove_duplicates(x) if isinstance(x, str) else x)
        df[column] = df[column].str.replace(r"\[|\]", "", regex=True)
    # to count the numbers in the _number_collected columns:
    for column in number_collected_columns:
        df[column + '_sum'] = df[column].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x != '' else 0)
    # creating csv and xlsx files
    df.to_csv(path_output + name_file_output_csv, index=False)
    df00 = df.applymap(lambda x: x.encode('unicode_escape').
                     decode('utf-8') if isinstance(x, str) else x)
    df00.to_excel(path_output + name_file_output_xlsx, engine="openpyxl")

    # Creating a new data frame to graph the results
    # Create a list of columns that end with '_number_collected_sum'
    number_collected_sum_columns = [col for col in df.columns if col.endswith('_number_collected_sum')]
    # Create a new DataFrame with just those columns
    df_graph = df[number_collected_sum_columns].copy()
    # Remove "_number_collected_sum" from the column names
    df_graph.columns = df_graph.columns.str.replace('_number_collected_sum', '')
    # Add the file_name column from the original DataFrame
    df_graph.insert(0, 'file_name', df['file_name'])
    # Sort the DataFrame by the fetaure_sort parameter
    df_graph = df_graph.sort_values(by=[feature_sort], ascending=False)  
    # creating csv and xlsx files for df_graph
    name_file_output_csv_graph = name_file_output_csv.replace(".csv", "_graph.csv")
    name_file_output_xlsx_graph = name_file_output_xlsx.replace(".xlsx", "_graph.xlsx")
    df_graph.to_csv(path_output + name_file_output_csv_graph, index=False)
    df00 = df_graph.applymap(lambda x: x.encode('unicode_escape').
                     decode('utf-8') if isinstance(x, str) else x)
    df00.to_excel(path_output + name_file_output_xlsx_graph, engine="openpyxl")
    # Make file_name' the index
    df_graph.set_index('file_name', inplace=True)
    # Create a heatmap
    heatmap_data = df_graph.head(number_articles).select_dtypes(include=[np.number])
    # Create the heatmap using seaborn
    plt.figure(figsize=(size_x, size_y))  # Adjust the figure size as necessary
    sns.heatmap(heatmap_data, annot=False, fmt=".1f", cmap='viridis')
    # Rotate the x-axis tick labels for better readability
    plt.xticks(rotation=45, ha='right')  
    # Rotate the y-axis tick labels for better readability
    plt.yticks(rotation=0)  
    # Show the heatmap
    plt.show()


def feature_pdf_extractor_table(directory,path_output, name_file_output_csv,name_file_output_xlsx):
    """
    
    """ 
    # Function to extract text from PDF file
    def extract_pdf_text(pdf_file):
        pdf_doc = fitz.open(pdf_file)
        text = ""
        for page in pdf_doc:
            text += page.get_text("text")
        pdf_doc.close()
        text = text.replace('\n', ' ')  # remove newline characters
        return text
    # Function to extract abstract from text from PDF file
    def extract_abstract(text):
        sentences = sent_tokenize(text)
        abstract = ""
        count = 0
        for sentence in sentences:
            if "abstract" in sentence.lower() or "a b s t r a c t" in sentence.lower():
                abstract = sentence + " "
                count += 1
            elif count > 0:
                abstract += sentence + " "
                count += 1
            if count == 10:
                break
        return abstract
    # Function to extract keywords from text from PDF file
    def extract_keywords(text):
        sentences = sent_tokenize(text)
        keywords = ""
        count = 0
        for sentence in sentences:
            if "keywords" in sentence.lower():
                words = word_tokenize(sentence)
                for word in words:
                    if count == 50:
                        break
                    if word.lower() not in ["keywords", ":"]:
                        keywords += word + " "
                        count += 1
        return keywords
    # Function to extract results from text from PDF file
    def extract_results(text):
        sentences = sent_tokenize(text)
        results = ""
        count = 0
        found = False
        for sentence in sentences:
            if "results" in sentence.lower() and not found:
                results = sentence + " "
                found = True
                count += 1
            elif found:
                results += sentence + " "
                count += 1
            if count == 100:
                break
        return results
    # Function to extract first two sentences from text from PDF file
    def extract_first_two_sentences(text):
        sentences = sent_tokenize(text)
        first_two = ""
        count = 0
        for sentence in sentences:
            first_two += sentence + " "
            count += 1
            if count == 2:
                break
        return first_two
    # Function to extract from sentence 3 to 14 from text from PDF file
    def extract_from_sentence_3_to_14(text):
        sentences = sent_tokenize(text)
        extracted = ""
        count = 0
        for sentence in sentences:
            if count >= 2 and count <= 13:
                extracted += sentence + " "
            count += 1
            if count == 14:
                break
        return extracted
    # Function to generate words dictionary
    def generate_words_dictionary(words):
        words_dictionary = {}
        for word in words:
            pattern = r'\b{}\b'.format(r'\s+'.join(word.split()))
            words_dictionary[word] = {
                'pattern': pattern,
                'context': []
            }
        return words_dictionary
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    # Create a list to store the data
    data = []
    column_names = ['file_name', 'title', 'abstract', 'results']
    # To sum the row values of the generated columns for each feature_word 
    def generate_feature_words(df):
        feature_words = []
        for row in range(0, df.shape[0]):
            feature_word = [df.iloc[row, 0]]
            feature_word_synonyms = df.iloc[row, 1:].dropna().tolist()
            feature_words.append((feature_word, feature_word_synonyms))
        return feature_words
    # Path to the feature file
    excel_features_file_path = "features.xlsx"
    # Read the Excel file into a DataFrame, considering the first row as data
    df_features = pd.read_excel(excel_features_file_path, header=None)
    # Generate feature words and their synonyms
    feature_words = generate_feature_words(df_features)
    # Extend column_names with dynamically generated column names for each feature_word and its synonyms
    for feature_word, feature_word_synonyms in feature_words:
        column_names.extend([f'{feature_word[0]}_present_{i+1}' for i in range(len(feature_word + feature_word_synonyms))])
        column_names.extend([f'{feature_word[0]}_context_{i+1}' for i in range(len(feature_word + feature_word_synonyms))])
    # Loop through the PDF files and apply the functions
    for pdf_file in pdf_files:
        words_to_detect_wrong_title = ['doi', 'No Job Name']
        pattern = '[0-9]{5,}'
        words_to_detect_wrong_title = [word for word in words_to_detect_wrong_title if not re.search(pattern, word)]
        file_path = os.path.join(directory, pdf_file)
        text = extract_pdf_text(file_path)
        pdf_doc = fitz.open(file_path)
        title = pdf_doc.metadata.get("title", "")
        abstract = extract_abstract(text)
        keywords = extract_keywords(text)
        results = extract_results(text)
        first_two = extract_first_two_sentences(text)
        from_sentence_3_to_14 = extract_from_sentence_3_to_14(text)
        title_improved = title if title and all(word not in title for word in words_to_detect_wrong_title) else first_two
        abstract_improved = abstract if abstract else from_sentence_3_to_14
        # Create a dictionary to store the data
        data_row = {
            'file_name': pdf_file,
            'title': title_improved,
            'abstract': abstract_improved,
            'results': results
        }
        # Loop through the feature_words and their synonyms
        for feature_word, feature_word_synonyms in feature_words:
            words_list = feature_word + feature_word_synonyms
            feature_word_and_synonyms = generate_words_dictionary(words_list)
            # Loop through the feature_word and its synonyms
            for i, (word, info) in enumerate(feature_word_and_synonyms.items()):
                pattern_word = re.compile(info['pattern'], re.IGNORECASE)
                matches = pattern_word.findall(text)
                count = len(matches)
                present = True if count > 0 else False
                if present:
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        if pattern_word.search(sentence):
                            info['context'].append(sentence)
            # Update the data_row with the generated columns for each feature_word and its synonyms
            data_row.update({f'{feature_word[0]}_present_{i+1}': len(info['context']) for i, (_, info) in enumerate(feature_word_and_synonyms.items())})
            data_row.update({f'{feature_word[0]}_context_{i+1}': info['context'] for i, (_, info) in enumerate(feature_word_and_synonyms.items())})
        # Append the data_row to the data list
        data.append(data_row)
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=column_names)
    # To sum the row values of the generated columns for each feature_word
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_present_')]
        df[f'{column_prefix}_present_sum'] = df[present_columns].sum(axis=1)
    # To sum the row values of the generated columns for each feature_word
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_context_')]
        df[f'{column_prefix}_context_sum'] = df[present_columns].sum(axis=1)
    # to delete duplicates in context_columns
    context_columns = [column_name for column_name in df.columns if column_name.endswith('_context_sum')]
    for column in context_columns:
        df[column] = df[column].apply(lambda x: list(set(x)) if isinstance(x, list) else [])
    # Sort according to the sum of present columns for each feature_word
    df = df.sort_values(by=[f'{feature_word[0]}_present_sum' for feature_word, _ in feature_words], ascending=False)
    # To sum the row values of the generated columns for each feature_word
    def generate_feature_unit_words(df):
        feature_unit_words = []
        for row in range(0, df.shape[0]):
            feature_unit_word = [df.iloc[row, 0]]
            feature_unit_word_synonyms = df.iloc[row, 1:].dropna().tolist()
            feature_unit_words.append((feature_unit_word, feature_unit_word_synonyms))
        return feature_unit_words
    # Path to the feature file
    excel_features_units_file_path = "features_units.xlsx"
    # Read the Excel file into a DataFrame, considering the first row as data
    df_features_units = pd.read_excel(excel_features_units_file_path, header=None)
    # Generate feature words and their synonyms
    feature_unit_words = generate_feature_unit_words(df_features_units)
    # Extend column_names with dynamically generated column names for each feature_unit_word and its synonyms
    def extract_number_1(lst, unit1):
        """Function to extract unique numbers integral or float followed by unit1, regardless of case"""
        if isinstance(lst, list):
            numbers = []
            for item in lst:
                if isinstance(item, str):
                    regex_list = [
                        # Pattern to extract a number followed by the unit: [number] unit
                        # match: '10 kg', '-3.14 m', '15.5Kg'. Extract: 10, -3.14, 15.5
                        fr'(-?\d+\.?\d*)\s*(?:{unit1})',
                        # Pattern to extract a number followed by ± number unit: [number]  ± number unit
                        # match: '20.5 ± 0.2 cm'. Extract: 20.5  
                        fr'(-?\d+(?:\.\d+)?)(?=\s*±\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to extract a number followed by (optional ,) and number  unit: [number] , number unit
                        # match: '5.2, and 8.9 cm' , '4.5 and 6 kg'. Extract: 5.2, 4.5
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] , number, and number unit 
                        # match: '1.1, 5.2, and 8.9 cm' , '1.2, 4.5 and 6 kg'. Extract: 1.1, 1.2
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, and number unit 
                        # match: '0.3, 1.1, 5.2, and 8.9 cm' , '0.5, 1.2, 4.5 and 6 kg'. Extract: 0.3, 0.5 
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, number,  and number unit 
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})',
                        # Pattern to ... [number] ,number, number, number, number,  and number unit
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})'
                        # Pattern to ... [number] ,number, number, number, number, number,  and number unit
                        fr'(-?\d+(?:\.\d+)?)(?=\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,?\s*and\s*\d+(?:\.\d+)?\s*{unit1})'
                    ]
                    numbers_set = set()
                    for regex in regex_list:
                        matches = re.findall(regex, item, re.IGNORECASE)
                        for match in matches:
                            if '.' in match:
                                numbers_set.add(float(match))
                            else:
                                numbers_set.add(int(match))
                    numbers_str = ', '.join(str(num) for num in numbers_set)
                    if numbers_str:
                        numbers.append(numbers_str)
                    else:
                        numbers.append("[]")
            return numbers
        else:
            return []
    # To put the numbers in the context columns
    for feature_unit_word, feature_unit_word_synonyms in feature_unit_words:
        prefix = feature_unit_word[0]
        prefix_context_sum = [column_name for column_name in df.columns if column_name.startswith(f'{prefix}_context_sum')]
        for unit1 in feature_unit_word_synonyms:
            df[f'{prefix}_context_number_{unit1}'] = df[prefix_context_sum].astype(str).apply(lambda x: extract_number_1(x.tolist(), unit1=unit1))
    # Collecting the numbers in the context_number columns
    for feature_word, _ in feature_words:# _ means ignore the second element
        column_prefix = feature_word[0]
        present_columns = [column_name for column_name in df.columns if column_name.startswith(f'{column_prefix}_context_number')]
        df[f'{column_prefix}_number_collected'] = df[present_columns].sum(axis=1)
    # Function to remove duplicates from a comma-separated string of numbers
    def remove_duplicates(s):
        # Split the string into a list of numbers, strip whitespace just in case
        numbers = [num.strip() for num in s.split(',')]
        # Convert the list to a set to remove duplicates, then back to a list
        unique_numbers = list(set(numbers))
        # Join the unique numbers back into a comma-separated string
        return ','.join(unique_numbers)
    # to delete "[]" in _number_collected columns:
    number_collected_columns = [column_name for column_name in df.columns if column_name.endswith('_number_collected')]
    for column in number_collected_columns:
        df[column] = df[column].apply(lambda x: remove_duplicates(x) if isinstance(x, str) else x)
        df[column] = df[column].str.replace(r"\[|\]", "", regex=True)
    # to count the numbers in the _number_collected columns:
    for column in number_collected_columns:
        df[column + '_sum'] = df[column].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x != '' else 0)
    # creating csv and xlsx files
    df.to_csv(path_output + name_file_output_csv, index=False)
    df00 = df.applymap(lambda x: x.encode('unicode_escape').
                     decode('utf-8') if isinstance(x, str) else x)
    df00.to_excel(path_output + name_file_output_xlsx, engine="openpyxl")

def feature_pdf_extractor_graph(path_output, name_file_output_csv,name_file_output_xlsx, feature_sort, number_articles, size_x, size_y, number, color):
    """
    
    """
    # Load the data from the CSV file into a DataFrame
    df = pd.read_csv(path_output + name_file_output_csv)
    # Creating a new data frame to graph the results
    # Create a list of columns that end with '_number_collected_sum'
    number_collected_sum_columns = [col for col in df.columns if col.endswith('_number_collected_sum')]
    # Create a new DataFrame with just those columns
    df_graph = df[number_collected_sum_columns].copy()
    # Remove "_number_collected_sum" from the column names
    df_graph.columns = df_graph.columns.str.replace('_number_collected_sum', '')
    # Add the file_name column from the original DataFrame
    df_graph.insert(0, 'file_name', df['file_name'])
    # Sort the DataFrame by the fetaure_sort parameter
    df_graph = df_graph.sort_values(by=[feature_sort], ascending=False)  
    # creating csv and xlsx files for df_graph
    name_file_output_csv_graph = name_file_output_csv.replace(".csv", "_graph.csv")
    name_file_output_xlsx_graph = name_file_output_xlsx.replace(".xlsx", "_graph.xlsx")
    df_graph.to_csv(path_output + name_file_output_csv_graph, index=False)
    df00 = df_graph.applymap(lambda x: x.encode('unicode_escape').
                     decode('utf-8') if isinstance(x, str) else x)
    df00.to_excel(path_output + name_file_output_xlsx_graph, engine="openpyxl")
    # Make file_name' the index
    df_graph.set_index('file_name', inplace=True)
    # Create a heatmap
    heatmap_data = df_graph.head(number_articles).select_dtypes(include=[np.number])
    heatmap_data = np.rint(heatmap_data).astype(int)
    # Create the heatmap using seaborn
    plt.figure(figsize=(size_x, size_y))  # Adjust the figure size as necessary
    sns.heatmap(heatmap_data, annot=number, fmt="d", cmap=color)
    # Rotate the x-axis tick labels for better readability
    plt.xticks(rotation=45, ha='right')  
    # Rotate the y-axis tick labels for better readability
    plt.yticks(rotation=0)  
    # Show the heatmap
    
    output_image_name = str(name_file_output_csv).replace('.csv', '_heatmap.png')
    # Rest of the code remains the same
    output_image_path = path_output
    output_image_file = output_image_path + output_image_name 
    plt.savefig(output_image_file, dpi=300)
    # show the plot
    plt.show()
    # Clear the figure to save memory
    plt.clf()
    plt.close()

def  feature_pdf_total_article_count(path_output, name_file_output_csv,y_label, x_label, title, color_bar, size_x, size_y):
    name_file_output_csv_graph = name_file_output_csv.replace(".csv", "_graph.csv")
    df_graph = pd.read_csv(path_output+name_file_output_csv_graph)
    df_graph.set_index('file_name', inplace=True)
    # Clean the DataFrame by removing rows with NaN values
    df_cleaned = df_graph.dropna()
    # Function that sets values to 1 if they are greater than 0
    def set_to_one(x):
        return 1 if x > 0 else x
    df_count_articles = df_cleaned.applymap(set_to_one)
     # Convert the Series to a DataFrame
    df_barchart = df_count_articles.sum().to_frame() 
    # Reset the index to get the column names as a column
    df_barchart.reset_index(inplace=True)
    # Rename the columns 
    df_barchart.columns = ['Feature', 'Total_count']
    # Figure size:
    plt.figure(figsize=(size_x, size_y)) 
    # Create the barplot
    ax = sns.barplot(data=df_barchart, y='Feature', x='Total_count', color= color_bar)
    # Set title and labels
    plt.title(title)
    plt.ylabel(y_label)  # Now corresponds to what was previously the x-axis
    plt.xlabel(x_label)  # Now corresponds to what was previously the y-axis
    # Annotate each bar with the value of its height (Total_count in this case)
    for p in ax.patches:
        ax.text(p.get_width(),  # X-position of the text
                p.get_y() + p.get_height() / 2,  # Y-position
                '{:.0f}'.format(p.get_width()),  # Text to display
                ha='left',  # Horizontal alignment
                va='center')  # Vertical alignment
    # Saving the plot
    output_image_name = str(name_file_output_csv).replace('.csv', '_article_count.png')
    # Rest of the code remains the same
    output_image_path = path_output
    output_image_file = output_image_path + output_image_name 

    plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    # Clear the figure to save memory
    plt.clf()
    plt.close()

