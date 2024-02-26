import subprocess
import sys
import pkg_resources

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required = {
    'Flask-JWT-Extended', 'streamlit', 'pandas', 'pysqlite3', 'regex',
    'numpy', 'collections-extended', 'nltk', 'matplotlib',
    'streamlit-authenticator', 'sympy', 'altair', 'sqlalchemy'
}

# Check if the required packages are installed, install if missing
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

for package in missing:
    install(package)

# NLTK stopwords additional setup
import nltk
nltk.download('stopwords')

from flask_jwt_extended import set_access_cookies
import streamlit as st
import pandas as pd
import sqlite3
import re
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import streamlit_authenticator as stauth
from sympy import comp
import altair as alt
#install ntlk bulgarian stopwords
from sqlalchemy import create_engine

global_df = None

SECRET_KEY = hash("open_sesame")
LOGGED_IN = False
def actual_streamlit_app():
    
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
    
    def job_market_growth(data):
        import datetime
        job_growth_df = data['date_added']
        
        def autocast_date(date):
            try:
                return datetime.datetime.strptime(date, '%d.%m.%Y')
            except:
                return datetime.datetime.now()

        job_growth_df = job_growth_df.dropna()
        job_growth_df = job_growth_df.apply(lambda x: autocast_date(x).date())
        job_growth_df = job_growth_df.value_counts().sort_index()
        job_growth_df = pd.DataFrame(job_growth_df)
        job_growth_df = job_growth_df.reset_index()
        job_growth_df.columns = ['Date', 'Job Count']
        
        st.altair_chart(alt.Chart(job_growth_df).mark_line().encode(
            x='Date',
            y='Job Count',
        ), use_container_width=True)
        st.write(job_growth_df)
            
    def score_compute(data,tokenized_text):
        if len(tokenized_text) == 0:
            st.write("Please enter a valid CV")
            return
        #extract all keywords, and compare them to the cv, remove all symbols, and just leave alphaneumerical characters
        tokenized_text = re.split(r'\W+', cv)
        tokenized_text = [word.lower() for word in tokenized_text]
        tokenized_text = [word for word in tokenized_text if word.isalpha()]
        #remove stop words
        stop_words = set(stopwords.words('english'))
        tokenized_text = [word for word in tokenized_text if word not in stop_words]
        set_tokenized_words = set(tokenized_text)
        st.write("Tokenized words: ",set_tokenized_words)
        #for entry in the table, generate a set of words and compare them to the set of tokenized words, and calculate the score
        data['text'] = data['text'].apply(lambda x: re.split(r'\W+', x))
        data['text'] = data['text'].apply(lambda x: [word.lower() for word in x])
        data['text'] = data['text'].apply(lambda x: [word for word in x if word.isalpha()])
        data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stop_words])
        data['text'] = data['text'].apply(lambda x: set(x))
        data['score'] = data['text'].apply(lambda x: len(x.intersection(set_tokenized_words)))
        st.write(data)
        return data
        
    def companies_sorted_by_highest_avg_score(data,tokenized_text):
        #run the keyword analysis on the data, and sort the companies by the average score on a plot 
        score_data = score_compute(data,tokenized_text)
        if score_data is None:
            return
        company_scores = {}
        #group the data by the company name and calculate the average score for each company
        unique_companies = score_data['secondary_text'].unique()
        for company in unique_companies:
            company_score = score_data[score_data['secondary_text'] == company]['score'].mean()
            company_scores[company] = company_score
        #transform the dict into a df
        company_scores = pd.DataFrame(list(company_scores.items()), columns=['company', 'average_score'])
        st.write(company_scores)
        #a bell curve of the average scores of the companies
        st.altair_chart(alt.Chart(company_scores).mark_bar().encode(
            x=alt.X(company_scores['average_score'].tolist(), title="Average Score"),
            y=alt.Y(company_scores['company'].to_list(), sort=None, title="Company"),
        ), use_container_width=True)
    def salary_distribution_analysis(data):
        salaries = []
        for matches in data['filtered_matches_from_text']:
            if matches is not None:
                for match in matches:
                    numbers = re.split(r'\D+', match)
                    for num in numbers:
                        if num:
                            salaries.append(int(num.replace(',', '')))
        # Create histogram data
        counts, bins = np.histogram(salaries, bins=30)
        bins = 0.5 * (bins[:-1] + bins[1:])  # Convert bin edges to centers
        st.bar_chart(pd.DataFrame({'Salary': bins, 'Count': counts}).set_index('Salary'))
        
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
        
                
        
    def company_salary_analysis(data):
        data['average_salary'] = data['filtered_matches_from_text'].apply(
            lambda x: np.mean([int(num.replace(',', '')) for match in x for num in re.split(r'\D+', match) if num]) if x else None)
        set_unique_companies = data['secondary_text'].unique()
        #for each unique company calculate the average salary
        company_salaries = {}
        for company in set_unique_companies:
            company_salaries[company] = data[data['secondary_text'] == company]['average_salary'].mean()
        
        #transform the dict into 2 columned df 
        company_salaries = pd.DataFrame(list(company_salaries.items()), columns=['company', 'average_salary'])
        
        company_salaries['average_salary'] = company_salaries['average_salary'].astype(float)
        company_salaries = company_salaries.sort_values(by='average_salary', ascending=False)
        
        st.altair_chart(alt.Chart(company_salaries).mark_bar().encode(
            x=alt.X('average_salary', title="Average Salary"),
            y=alt.Y('company', sort=None, title="Company"),
        ), use_container_width=True)
        st.write(company_salaries)
        
        
        
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
        
        most_common_words['Frequency'] = most_common_words['Frequency'].astype(int)
        most_common_words['Word'] = most_common_words['Word'].astype(str)

        st.write(most_common_words)

    # Function to load data
    def load_data():
        #ask the user if they want a remote database or not 
        remote_db = st.checkbox('Use Remote Database')
        if remote_db:
            #prompt the user for the database credentials
            user = st.text_input('Username')
            password = st.text_input('Password', type='password')
            host = st.text_input('Host')
            port = st.text_input('Port')
            database = st.text_input('Database')
            # Create a connection string
            #on click of a button, try connecting with the given credentials
            if st.button('Connect'):    
                # This connection string is for MySQL databases
                conn = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'

                # Create an engine to connect to the MySQL server
                engine = create_engine(conn)

                # Query to select data (modify as needed)
                conn = sqlite3.connect('jobs.db')
                temp = pd.read_sql_query('SELECT * FROM JobPosts', engine)
                temp = temp.merge(pd.read_sql_query('SELECT * FROM longevity_tracker', engine), on='fingerprint',how='left')
                temp = temp.merge(pd.read_sql_query('SELECT fingerprint, view_count, date, MAX(idx) as max_idx FROM view_counts WHERE view_count != \'Nan\'  GROUP BY fingerprint, view_count, date', engine), on='fingerprint',how='left')
                #write any errors to the user
                global_df = temp
                return temp
        else:      
            conn = sqlite3.connect('jobs.db')
            temp = pd.read_sql_query('SELECT * FROM JobPosts', conn)
            temp = temp.merge(pd.read_sql_query('SELECT * FROM longevity_tracker', conn), on='fingerprint',how='left')
            temp = temp.merge(pd.read_sql_query('SELECT * FROM view_counts WHERE view_count!="Nan" GROUP BY fingerprint having max(view_count)', conn), on='fingerprint',how='left')
            global_df = temp
            return temp


    st.set_page_config(layout='wide')  # Set to wide mode
    df = load_data()
    
    #destroy the button
    st.empty()
    
    # Streamlit application layout
    st.title('Job Listings Interface')

    st.sidebar.header('Advanced Filter\n Columns:')
    columns = df.columns.tolist()
    #streamlit seperator
    
    # Dynamic generation of input fields based on user selection
    user_queries = {}
    for column in columns:
        # Let user decide which columns to query
        if st.sidebar.checkbox(f'{column}', key=f'chk_{column}'):
            # Allow multiple inputs for each field
            query_input = st.sidebar.text_input(f'(comma,seperated) <empty> = null:', key=f'in_{column}')
            user_queries[column] = query_input.split(',')

    st.sidebar.markdown('---')
    
    # Function to filter dataframe based on user queries
    def filter_dataframe(df, queries):
        #return the dataframe, return the original if none are provided, and if any are provided, return a filtered df
        if not queries:
            return df
        else:
            for key in user_queries.keys():
                for query in user_queries[key]:
                    if query:
                        #check if the user is trying to target empty cells or not
                        if query == '<empty>':
                            df = df[df[key].isnull()]
                        else:
                            df = df[df[key].str.contains(query)]    
            return df
        

    # Filter data based on user queries
    df_filtered = filter_dataframe(df, user_queries)
    
    # Display the DataFrame, paged
    
    st.dataframe(df_filtered, height=600)

    # Analysis options
    analysis_options = ['Salary Distribution Analysis', 'Job Title Salary Analysis', 
                        'Company Salary Analysis', 'Frequency of Words Analysis', 'Job Market Growth Analysis','Score based on given keywords','Companies sorted by highest average score']
    analysis_choice = st.sidebar.selectbox('Select Analysis Type', analysis_options)

    if analysis_choice == 'Frequency of Words Analysis':
        # Parameter inputs for analysis
        len_of_min_word = st.sidebar.slider('Minimum word length for frequency analysis', 3, 10, 3)
        most_common = st.sidebar.slider('Number of most common words to display', 50, 200, 100)
        
    st.sidebar.markdown('---')
    # Choose dataset for analysis
    dataset_choice = st.sidebar.radio('Choose Dataset for Analysis', ('Complete Dataset', 'Filtered Dataset'))
    data_to_analyze = df if dataset_choice == 'Complete Dataset' else df_filtered
    if analysis_choice in ['Score based on given keywords','Companies sorted by highest average score']:
        cv = st.sidebar.text_input("Enter your CV here")
        tokenized_text = re.split(r'\W+', cv)
        tokenized_text = [word.lower() for word in tokenized_text]
        tokenized_text = [word for word in tokenized_text if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokenized_text = [word for word in tokenized_text if word not in stop_words]
        
        
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
        elif analysis_choice == 'Job Market Growth Analysis':
            job_market_growth(data_to_analyze)
        elif analysis_choice == 'Score based on given keywords':
            score_compute(data_to_analyze,tokenized_text)
        elif analysis_choice == 'Companies sorted by highest average score':
            companies_sorted_by_highest_avg_score(data_to_analyze,tokenized_text)

actual_streamlit_app()

