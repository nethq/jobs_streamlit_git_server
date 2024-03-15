import os
import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np
import altair as alt
from sqlalchemy import create_engine
import datetime
from datetime import datetime
import time

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"} | {"и", "в", "во", "за", "со", "од", "се", "да", "не", "ние", "вие", "те", "тя", "то", "техни", "тях", "мой", "твой", "свой", "наш", "ваш", "свой", "мои", "твои", "свои", "наши", "ваши", "твои", "моите", "твоите", "своите", "нашите", "вашият", "твоят", "своят", "моето", "твоето", "своето", "нашето", "вашият", "твоите", "своите", "нашите", "вашия", "твоя", "своя", "моето", "твоето", "своето", "нашето", "вашата", "твоята", "своята", "моят", "твоят", "своят", "моите", "твоите", "своите", "нашите", "вашият", "твоя", "своя", "моето", "твоето", "своето", "нашето", "вашия", "твоя", "своя", "моите", "твоите", "своите", "нашите", "вашия", "твоя", "своя", "моето", "твоето", "своето", "нашето", "вашия", "твоя", "своя", "какво", "което", "когато", "къде", "защо", "как", "само", "от", "сам", "сама", "сами", "само", "все", "всичко", "някой", "някакъв", "някаква", "няколко", "малко", "много", "малък", "голям", "добър", "лош", "нов", "стар", "свой", "мъртъв", "жив", "слаб", "силен", "голям", "малък", "стар", "нов", "добър", "лош", "цял", "целия", "цялата", "цяло", "целият", "цялото", "друг", "друга", "друго", "други", "един", "едно", "една", "едни", "всички", "няколко", "който", "какъв", "каква", "какво", "кой", "коя", "кои", "които", "откъде", "докъде", "кога", "как", "кои", "които", "кой", "какъв", "каква", "какво", "всичко", "няколко", "малко", "много", "доста", "най", "малко", "много", "малко", "само", "няколко", "сам", "сама", "само", "нищо", "сичко", "нещо", "няколко", "всички", "друг", "друга", "друго", "други", "един", "едно", "една", "едни", "всички", "няколко", "който", "какъв", "каква", "какво", "кой", "коя", "кои", "които", "откъде", "докъде", "кога", "как", "колко", "къде", "защо", "кой", "коя", "кои"}

st.set_page_config(layout='wide')

def show_popup(prompt="Enter text here:"):
    pass
    # result = st.text_input(prompt)
    # if st.button("OK") and result:
    #     return result
    # else:
    #     st.stop()

def connect_to_database(remote_db, user, password, host, port, database):
    if remote_db:
        conn = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'
        return create_engine(conn)
    else:
        return None
        pass

# Function to load data from database    
@st.cache_data
def load_data(_engine,time_hash=None):
    try:
        query = 'SELECT * FROM JobPosts'
        temp = pd.read_sql_query(query, _engine)
        temp = temp.merge(pd.read_sql_query('SELECT * FROM longevity_tracker', _engine), on='fingerprint', how='left')
        temp = temp.merge(pd.read_sql_query('SELECT fingerprint, MAX(view_count) AS highest_view_count, count(fingerprint) as observations FROM view_counts GROUP BY fingerprint', _engine), on='fingerprint', how='left')
        return temp
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def dynamic_execution(session_state,_engine):
    df = pd.read_sql_query('SELECT * FROM py_lib', _engine)
    session_state.dynamic_functions_df = df
    try:        
        for index, row in df.iterrows():
            try:
                if st.button(f"{row['idx']}-{row['fn_name']}"):
                        exec(row['exec_code'], globals(), locals())
            except Exception as e:
                st.error(f"Error running singular dynamic execution function: {e}")
    except Exception as e:
        st.error(f"Error running dynamic execution: {e}")

# Function to filter dataframe based on user queries
def filter_dataframe(df, queries):
    if not queries:
        return df
    for key in queries.keys():
        #print the type of the value
        if queries[key] == "" or queries[key] is None:
            continue
        if "," in queries[key]:
            sub_terms = queries[key].split(",")
            for sub_term in sub_terms:
                if sub_term:
                    df = advanced_text_filtering(df, key, sub_term)
        else:
            df = advanced_text_filtering(df, key, queries[key])
    return df

def advanced_text_filtering(df, key, query):
    if "|" in query:
        terms = query.split("|")
        st.warning("OR "+str(terms))
        filtered_df = df[df[key].apply(lambda x: terms[0] in x or terms[1] in x)]
    elif "&" in query:
        terms = query.split("&")
        st.warning("AND "+str(terms))
        filtered_df = df[df[key].apply(lambda x: terms[0] in x and terms[1] in x)]
    elif "!" in query:
        #it will be "!<term>", so you need to further filter for the negation
        term = query[1:]
        st.warning("NOT "+str(term))
        filtered_df = df[df[key].apply(lambda x: term not in x)]
    elif query == "<empty>":
        #return the df where the values in this column are Null/NaN
        st.warning("EMPTY")
        filtered_df = df[df[key].isnull()]
    else:
        st.warning("SINGLE "+str(query))
        filtered_df = df[df[key].apply(lambda x: query in x)]
        
    return filtered_df

def salary_extract_df(dataframe, minimum_wage=900, maximum_salary=1000000):
    def extract_sequences(input_string):
        # Simplified pattern to capture numerical values possibly in a range
        pattern = r'\b\d+\s*(?:до|-)?\s*\d*'
        
        # Find all matches using the pattern
        matches = re.findall(pattern, input_string)

        # Process matches to extract numerical values
        processed_matches = []
        for match in matches:
            # Find the starting index of each match
            start_index = input_string.find(match)
            # Check for "BGN" or "EUR" within 25 characters of the match
            pre_text = input_string[max(0, start_index-25):start_index]
            post_text = input_string[start_index:start_index+25]
            if "BGN" in pre_text or "BGN" in post_text or "EUR" in pre_text or "EUR" in post_text:
                # Extract numbers from the match
                nums = re.findall(r'\d+', match)
                # Filter and convert numbers, checking against min and max thresholds
                for num in nums:
                    num_int = int(num)
                    if minimum_wage <= num_int <= maximum_salary:
                        processed_matches.append(num_int)
        
        return processed_matches

    def calculate_average_salary(salaries):
        if not salaries:
            return None
        # Compute the average of the provided salary numbers
        return sum(salaries) / len(salaries)

    # Updated function to include the refined salary calculation
    for index, row in dataframe.iterrows():
        text_matches = extract_sequences(row['text'])
        card_info_matches = extract_sequences(row['card_info'])
        
        # Combine all numerical matches from both text and card_info
        all_matches = text_matches + card_info_matches
        
        # Store all found salary numbers for reference
        dataframe.at[index, 'matches'] = str(all_matches)
        
        # Calculate average salary from all extracted numbers, if below minimum_wage set to None
        average_salary = calculate_average_salary(all_matches)
        if minimum_wage and average_salary and average_salary < minimum_wage:
            average_salary = None
        dataframe.at[index, 'average_salary'] = average_salary

    return dataframe


def extract_potential_salaries(text):
    
    matches = re.findall(r"(\d+.{1,5}\d+)", str(text))
    filtered_matches = []
    for match in matches:
        start_idx = text.find(match)
        end_idx = start_idx + len(match)
        surrounding_text = text[max(0, start_idx - 20):min(end_idx + 20, len(text))]
        if "BGN" in surrounding_text or "EUR" in surrounding_text:
            filtered_matches.append(match)
    return filtered_matches if filtered_matches else None

# Function to perform score computation based on keywords
def score_compute(data, tokenized_text):
    if isinstance(tokenized_text, str):
        tokenized_text = set(tokenized_text.lower().split())
    elif isinstance(tokenized_text, set):
        tokenized_text = {word.lower() for word in tokenized_text}

    if isinstance(data['text'], str):
        data['score'] = data['text'].apply(lambda x: len(set(re.split(r'\W+', x.lower())).difference(stop_words).intersection(tokenized_text)))        
    else:
        data['score'] = data['text'].apply(lambda x: len(set(re.split(r'\W+', x.lower())).difference(stop_words).intersection(tokenized_text)) if isinstance(x, str) else 0)
    st.write("Dataframe including scores:")
    st.dataframe(data)
    return data

# Function to perform analysis on companies sorted by highest average score
def companies_sorted_by_highest_avg_score(data, tokenized_text):
    score_data = score_compute(data, tokenized_text)
    if score_data is None:
        return
    company_scores = {}
    unique_companies = score_data['secondary_text'].unique()
    posts_count = {}
    for company in unique_companies:
        posts_count[company] = score_data[score_data['secondary_text'] == company].shape[0]
        company_score = score_data[score_data['secondary_text'] == company]['score'].mean()
        company_scores[company] = company_score
    company_scores = pd.DataFrame(list(company_scores.items()), columns=['company', 'average_score'])
    company_scores['posts_count'] = company_scores['company'].apply(lambda x: posts_count[x])
    if isinstance(company_scores, pd.DataFrame):
        company_scores = company_scores.sort_values(by=['average_score', 'posts_count'], ascending=False)
    st.write(company_scores)
    st.altair_chart(alt.Chart(company_scores).mark_bar().encode(
        x=alt.X('average_score', title="Average Score"),
        y=alt.Y('company', sort=None, title="Company"),
    ), use_container_width=True)

# Function to perform salary distribution analysis
def salary_distribution_analysis(data):
    if 'average_salary' not in data.columns:
        salary_extract_df(data)
    salaries = data['average_salary'].tolist()
    salaries = [salary for salary in salaries if salary is not None]
    counts, bins = np.histogram(salaries, bins=30)
    bins = 0.5 * (bins[:-1] + bins[1:])
    st.bar_chart(pd.DataFrame({'Salary': bins, 'Count': counts}).set_index('Salary'))
    st.write(pd.DataFrame({'Salary': bins, 'Count': counts}))

# Function to perform job title salary analysis
def job_title_salary_analysis(data):
    if 'average_salary' not in data.columns:
        salary_extract_df(data)
    set_unique_titles = data['card_title'].unique()
    title_salaries = {}
    for title in set_unique_titles:
        title_salaries[title] = data[data['card_title'] == title]['average_salary'].mean()
    title_salaries = pd.DataFrame(list(title_salaries.items()), columns=['title', 'average_salary'])
    title_salaries['average_salary'] = title_salaries['average_salary'].astype(float)
    title_salaries = title_salaries.sort_values(by='average_salary', ascending=False)
    st.altair_chart(alt.Chart(title_salaries).mark_bar().encode(
        x=alt.X('average_salary', title="Average Salary"),
        y=alt.Y('title', sort=None, title="Job Title"),
    ), use_container_width=True)
    #write the counts of each job title's job posts with salary
    title_salaries['posts_count'] = title_salaries['title'].apply(lambda x: data[data['card_title'] == x].shape[0])
    st.write(title_salaries)

# Function to perform company salary analysis
def company_salary_analysis(data):
    if 'average_salary' not in data.columns:
        salary_extract_df(data)
    set_unique_companies = data['secondary_text'].unique()
    company_salaries = {}
    for company in set_unique_companies:
        company_salaries[company] = data[data['secondary_text'] == company]['average_salary'].mean()
    company_salaries = pd.DataFrame(list(company_salaries.items()), columns=['company', 'average_salary'])
    company_salaries['average_salary'] = company_salaries['average_salary'].astype(float)
    company_salaries = company_salaries.sort_values(by='average_salary', ascending=False)
    st.altair_chart(alt.Chart(company_salaries).mark_bar().encode(
        x=alt.X('average_salary', title="Average Salary"),
        y=alt.Y('company', sort=None, title="Company"),
    ), use_container_width=True)
    #add the counts of each company's job posts with salary
    company_salaries['posts_count'] = company_salaries['company'].apply(lambda x: data[data['secondary_text'] == x].shape[0])
    st.write(company_salaries)

def normalize_and_display_scores(dataframe):
    # Calculate the average score
    average_score = dataframe['score'].mean()
    
    # Normalize scores by dividing by the average score
    dataframe['normalized_score'] = dataframe['score'] / average_score
    
    # Sort the DataFrame by descending normalized score
    sorted_dataframe = dataframe[['fingerprint', 'secondary_text', 'normalized_score']].sort_values(by='normalized_score', ascending=False)
    
    # Display the sorted DataFrame
    st.write(sorted_dataframe)
    
    # Calculate the mean score for each secondary_text
    mean_scores = sorted_dataframe.groupby('secondary_text').agg(mean_normalized_score=('normalized_score', 'mean')).reset_index()
    
    # Create Altair chart
    chart = alt.Chart(mean_scores).mark_bar().encode(
        x=alt.X('mean_normalized_score:Q', title="Mean Normalized Score"),
        y=alt.Y('secondary_text:N', sort='-x', title="Company"),  # Sort by mean_normalized_score in descending order
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

def frequency_of_words_analysis(data, len_of_min_word=3, most_common=100):
    #prompt the user to select if they wish to check a company's job postings, how common is their wording -
    words = []
    for text in data['text']:
        words.extend([word for word in text.split() if word not in stop_words])
    word_counter = Counter(words)
    for word in list(word_counter):
        if len(word) < len_of_min_word:
            del word_counter[word]
    most_common_words = pd.DataFrame(word_counter.most_common(most_common), columns=['Word', 'Frequency']).set_index('Word')
    st.write(most_common_words)
      
# Streamlit UI
def main():
    session_state = st.session_state
    
    # Initialize session state variables
    if "dynamic_functions_df" not in session_state:
        session_state.dynamic_functions_df = None
    if "sync" not in session_state:
        session_state.sync = False
    if 'db_engine' not in session_state:
        session_state.db_engine = None
    if 'original_df' not in session_state:
        session_state.original_df = None
    if 'df_filtered' not in session_state:
        session_state.df_filtered = None
    if 'applied_filters' not in session_state:
        session_state.applied_filters = None

    # Check if database connection exists
    if session_state.db_engine is None:
        st.sidebar.header('Database Connection')
        remote_db = st.sidebar.checkbox('Use Remote Database')
        if remote_db:
            user = st.sidebar.text_input('Username')
            password = st.sidebar.text_input('Password', type='password')
            host = st.sidebar.text_input('Host')
            port = st.sidebar.text_input('Port')
            database = st.sidebar.text_input('Database')
            if st.sidebar.button('Connect'):
                session_state.db_engine = connect_to_database(remote_db, user, password, host, port, database)
                if session_state.db_engine:
                    st.sidebar.success("Database connection successful!")
                else:
                    st.sidebar.error("Database connection failed!")
        else:
            st.sidebar.info("No database connection needed for local data processing.")

    # Check if DataFrame is loaded
    if session_state.original_df is None and session_state.db_engine:
        st.sidebar.header('Data Loading')
        if st.sidebar.button('Load Data'):
            session_state.original_df = load_data(session_state.db_engine,datetime.now().strftime("%Y-%m-%d %H"))
            dynamic_execution(session_state,session_state.db_engine)
        if st.sidebar.button("Reload Data"):
            session_state.original_df = load_data(session_state.db_engine,datetime.now())
            dynamic_execution(session_state,session_state.db_engine)
    elif session_state.original_df is None and not session_state.db_engine:
        st.sidebar.error("No database connection available to load data from!")
        st.stop()
        
    # Analysis options
    analysis_options = ['Salary Distribution Analysis', 'Job Title Salary Analysis', 
                        'Company Salary Analysis', 'Frequency of Words Analysis', 'Score based on given keywords',
                        'Companies sorted by highest average score','Normalized Scores Against CV']
    
    analysis_choice = st.sidebar.selectbox('Select Analysis Type', analysis_options)

    if analysis_choice == 'Frequency of Words Analysis':
        # Parameter inputs for analysis
        len_of_min_word = st.sidebar.slider('Minimum word length for frequency analysis', 3, 10, 3)
        most_common = st.sidebar.slider('Number of most common words to display', 50, 200, 100)
        
    st.sidebar.markdown('---')

    # Perform filtering based on user input
    queries = {}
    if session_state.df_filtered is not None:
        session_state.df_filtered = None  # Reset filtered DataFrame before applying new filters
    
    dataset_choice = st.sidebar.radio('Choose Dataset for Analysis', ('Complete Dataset', 'Filtered Dataset'))
    if dataset_choice == 'Filtered Dataset':
        try:            
            session_state.display_complete_dataset = False
            for key in session_state.original_df.columns:
                queries[key] = st.sidebar.text_input(f'Filter by {key}', key=f'filter_{key}')
            session_state.df_filtered = filter_dataframe(session_state.original_df, queries)
            session_state.applied_filters = queries
        except Exception as e:
            st.error(f"Error applying filters: {e}")
    else:
        session_state.display_complete_dataset = True
        session_state.applied_filters = None
        
        # Display loaded DataFrame
    if session_state.original_df is not None:
        #allow the user to input a number to identify how many records to show, as to not slow down the app,
        num_records = st.sidebar.number_input('Number of records to display', min_value=1, max_value=session_state.original_df.shape[0], value=1000, step=100)
        if session_state.applied_filters:
            st.write("Filtered Dataset, Applied Filters:")
            #display in a small table
            #show only the non-empty display filters
            #the displayfilters is a dictionary with the following structure - {'column_name': str('filter_string')}
            display_filters = {k: v for k, v in session_state.applied_filters.items() if v}

            st.write(pd.DataFrame(display_filters.items(), columns=['Filter', 'Value']).set_index('Filter'))
            st.write("Number of records after filtering: ", session_state.df_filtered.shape[0])
            if num_records <= session_state.df_filtered.shape[0]:
                st.dataframe(session_state.df_filtered.head(num_records), height=600)
            else:
                st.dataframe(session_state.df_filtered, height=600)
            session_state.selected_df = session_state.df_filtered
        else:
            if num_records <= session_state.original_df.shape[0]:
                st.write("Complete dataset, no filters applied.")
                st.write("Number of records: ", session_state.original_df.shape[0])
                st.dataframe(session_state.original_df.head(num_records), height=600)
            else:    
                st.write("Complete dataset: ", session_state.original_df.shape[0], "total records.")
                st.dataframe(session_state.original_df, height=600)
            session_state.selected_df = session_state.original_df


    data_to_analyze = session_state.original_df if dataset_choice == 'Complete Dataset' else session_state.df_filtered
        
    if analysis_choice in ['Score based on given keywords','Companies sorted by highest average score', 'Normalized Scores Against CV']:
        cv = st.sidebar.text_area("Enter text / keywords", height=200)
        
    if session_state.db_engine and session_state.dynamic_functions_df is not None:       
        dynamic_execution(session_state,session_state.db_engine)
        

    # Perform analysis based on user choice
    if st.sidebar.button('Run Analysis'):
        # Extract potential salaries for analysis
        try:                
            # Analysis based on user selection
            if analysis_choice == 'Salary Distribution Analysis':
                salary_distribution_analysis(salary_extract_df(data_to_analyze))
            elif analysis_choice == 'Job Title Salary Analysis':
                job_title_salary_analysis(salary_extract_df(data_to_analyze))
            elif analysis_choice == 'Company Salary Analysis':
                company_salary_analysis(salary_extract_df(data_to_analyze))
            elif analysis_choice == 'Frequency of Words Analysis':
                frequency_of_words_analysis(data_to_analyze, len_of_min_word, most_common)
            elif analysis_choice == 'Score based on given keywords':
                tokenized_text = re.split(r'\W+', cv)
                tokenized_text = [word.lower() for word in tokenized_text]
                tokenized_text = [word for word in tokenized_text if word.isalpha()]
                tokenized_text = [word for word in tokenized_text if word not in stop_words]
                score_compute(data_to_analyze, tokenized_text)
            elif analysis_choice == 'Companies sorted by highest average score':
                tokenized_text = re.split(r'\W+', cv)
                tokenized_text = [word.lower() for word in tokenized_text]
                tokenized_text = [word for word in tokenized_text if word.isalpha()]
                tokenized_text = [word for word in tokenized_text if word not in stop_words]
                companies_sorted_by_highest_avg_score(data_to_analyze, tokenized_text)
            elif analysis_choice == 'Normalized Scores Against CV':
                score_compute(data_to_analyze, cv)
                if "score" in data_to_analyze.columns:
                    normalize_and_display_scores(data_to_analyze)
                else:
                    st.error("No scores found in the dataset. Please run the score based on given keywords analysis first.")
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            
if __name__ == "__main__":
    main()
