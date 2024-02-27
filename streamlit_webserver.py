import os
import streamlit as st
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import altair as alt
from sqlalchemy import create_engine


# Function to establish database connection
def connect_to_database(remote_db=False, user=None, password=None, host=None, port=None, database=None):
    if remote_db:
        conn = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'
        return create_engine(conn)
    else:
        # Connect to local database or any other preferred method
        pass

# Function to load data from the database
@st.cache_data
def load_data(_engine):
    try:
        query = 'SELECT * FROM JobPosts'
        temp = pd.read_sql_query(query, _engine)
        temp = temp.merge(pd.read_sql_query('SELECT * FROM longevity_tracker', _engine), on='fingerprint', how='left')
        temp = temp.merge(pd.read_sql_query('SELECT fingerprint, view_count, date, MAX(idx) as max_idx FROM view_counts WHERE view_count != \'Nan\'  GROUP BY fingerprint, view_count, date', _engine), on='fingerprint', how='left')
        return temp
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to filter dataframe based on user queries
def filter_dataframe(df, queries):
    if not queries:
        return df
    else:
        for key in queries.keys():
            for query in queries[key]:
                if query:
                    if query == '<empty>':
                        df = df[df[key].isnull()]
                    else:
                        df = advanced_text_filtering(df, key, query)
        return df

# Function to perform advanced text filtering
def advanced_text_filtering(df, key, query):
    processed_groups = []
    for group in query.split(')'):
        processed_group = []
        for part in group.split('('):
            if '|' in part:
                processed_group.append('|'.join(f'({token.strip()})' for token in part.split('|')))
            else:
                processed_group.append(part.strip())
        processed_groups.append('&'.join(processed_group))
    final_query = '|'.join(processed_groups)
    return df[df[key].str.contains(final_query, na=False)]

# Function to extract potential salaries from text
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

# Function to compute score based on keywords
def score_compute(data, tokenized_text):
    
    if len(tokenized_text) == 0:
        st.write("Please enter a valid CV")
        return
    tokenized_text = re.split(r'\W+', cv)
    tokenized_text = [word.lower() for word in tokenized_text]
    tokenized_text = [word for word in tokenized_text if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokenized_text = [word for word in tokenized_text if word not in stop_words]
    set_tokenized_words = set(tokenized_text)
    data['text'] = data['text'].apply(lambda x: re.split(r'\W+', x))
    data['text'] = data['text'].apply(lambda x: [word.lower() for word in x])
    data['text'] = data['text'].apply(lambda x: [word for word in x if word.isalpha()])
    data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stop_words])
    data['text'] = data['text'].apply(lambda x: set(x))
    data['score'] = data['text'].apply(lambda x: len(x.intersection(set_tokenized_words)))
    return data

# Function to perform analysis on companies sorted by highest average score
def companies_sorted_by_highest_avg_score(data, tokenized_text):
    score_data = score_compute(data, tokenized_text)
    if score_data is None:
        return
    company_scores = {}
    unique_companies = score_data['secondary_text'].unique()
    for company in unique_companies:
        company_score = score_data[score_data['secondary_text'] == company]['score'].mean()
        company_scores[company] = company_score
    company_scores = pd.DataFrame(list(company_scores.items()), columns=['company', 'average_score'])
    st.write(company_scores)
    st.altair_chart(alt.Chart(company_scores).mark_bar().encode(
        x=alt.X('average_score', title="Average Score"),
        y=alt.Y('company', sort=None, title="Company"),
    ), use_container_width=True)

# Function to perform salary distribution analysis
def salary_distribution_analysis(data):
    salaries = []
    for matches in data['filtered_matches_from_text']:
        if matches is not None:
            for match in matches:
                numbers = re.split(r'\D+', match)
                for num in numbers:
                    if num:
                        salaries.append(int(num.replace(',', '')))
    counts, bins = np.histogram(salaries, bins=30)
    bins = 0.5 * (bins[:-1] + bins[1:])
    st.bar_chart(pd.DataFrame({'Salary': bins, 'Count': counts}).set_index('Salary'))

# Function to perform job title salary analysis
def job_title_salary_analysis(data):
    data['average_salary'] = data['filtered_matches_from_text'].apply(
        lambda x: np.mean([int(num.replace(',', '')) for match in x for num in re.split(r'\D+', match) if num]) if x else None)
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
    st.write(title_salaries)

# Function to perform company salary analysis
def company_salary_analysis(data):
    data['average_salary'] = data['filtered_matches_from_text'].apply(
        lambda x: np.mean([int(num.replace(',', '')) for match in x for num in re.split(r'\D+', match) if num]) if x else None)
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
    st.write(company_salaries)

# Function to perform frequency of words analysis
def frequency_of_words_analysis(data, len_of_min_word=3, most_common=100):
    stop_words = stopwords.words('english')
    stop_words.extend(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
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
    st.set_page_config(layout='wide')
    session_state = st.session_state

    # Initialize session state variables
    if 'db_engine' not in session_state:
        session_state.db_engine = None
    if 'df' not in session_state:
        session_state.df = None
    if 'df_filtered' not in session_state:
        session_state.df_filtered = None

    st.title('Job Listings Interface')

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
    if session_state.df is None and session_state.db_engine:
        st.sidebar.header('Data Loading')
        if st.sidebar.button('Load Data'):
            session_state.df = load_data(session_state.db_engine)
            if session_state.df is not None:
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.error("Failed to load data!")

    # Display loaded DataFrame
    if session_state.df is not None:
        if session_state.df_filtered is not None:
            dataset_choice = st.sidebar.radio('Choose Dataset for Analysis', ('Complete Dataset', 'Filtered Dataset'))
            if dataset_choice == 'Filtered Dataset':
                st.dataframe(session_state.df_filtered, height=600)
            else:
                st.dataframe(session_state.df, height=600)
        else:
            st.dataframe(session_state.df, height=600)

    # Analysis options
    analysis_options = ['Salary Distribution Analysis', 'Job Title Salary Analysis', 
                        'Company Salary Analysis', 'Frequency of Words Analysis', 'Score based on given keywords','Companies sorted by highest average score']
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
        for key in session_state.df.columns:
            queries[key] = [st.sidebar.text_input(f'Filter by {key}', key=f'filter_{key}')]
        session_state.df_filtered = filter_dataframe(session_state.df, queries)
    data_to_analyze = session_state.df if dataset_choice == 'Complete Dataset' else session_state.df_filtered
        
    if analysis_choice in ['Score based on given keywords','Companies sorted by highest average score']:
        cv = st.sidebar.text_input("Enter your CV here")
        
        
    # Perform analysis based on user choice
    if st.sidebar.button('Run Analysis'):
        # Extract potential salaries for analysis
        data_to_analyze['filtered_matches_from_text'] = data_to_analyze['text'].apply(extract_potential_salaries)
        
        # Analysis based on user selection
        if analysis_choice == 'Salary Distribution Analysis':
            salary_distribution_analysis(data_to_analyze)
        elif analysis_choice == 'Job Title Salary Analysis':
            job_title_salary_analysis(data_to_analyze)
        elif analysis_choice == 'Company Salary Analysis':
            company_salary_analysis(data_to_analyze)
        elif analysis_choice == 'Frequency of Words Analysis':
            frequency_of_words_analysis(data_to_analyze, len_of_min_word, most_common)
        elif analysis_choice == 'Score based on given keywords':
            #get the tokenized text
            tokenized_text = re.split(r'\W+', cv)
            tokenized_text = [word.lower() for word in tokenized_text]
            tokenized_text = [word for word in tokenized_text if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokenized_text = [word for word in tokenized_text if word not in stop_words]
            score_compute(data_to_analyze, tokenized_text)
        elif analysis_choice == 'Companies sorted by highest average score':
            tokenized_text = re.split(r'\W+', cv)
            tokenized_text = [word.lower() for word in tokenized_text]
            tokenized_text = [word for word in tokenized_text if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokenized_text = [word for word in tokenized_text if word not in stop_words]
            companies_sorted_by_highest_avg_score(data_to_analyze, tokenized_text)

if __name__ == "__main__":
    main()

