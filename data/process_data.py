# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:08:45 2021


@author: Tino Riebe
"""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT  
    messages_filepath(string) - Filepath of Disaster_messgases.csv 
    categories_filepath(string) - Filepath of Disaster_category.csv
    
    OUTPUT
    df - complete dataframe    
    '''
    df_msg = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df = pd.merge(df_msg, df_cat, on="id")
    return df


def clean_data(df):
    '''
    Clean Data :
    1. Clean and Transform Category Columns from categories csv
    2. Drop Duplicates
    3. Remove any missing values
        
    INPUT
    df - complete dataframe
    
    OUTPUT
    df - cleaned dataframe
    '''
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    
    # Get new column names from category columns
    # category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    categories = categories.applymap(lambda s: int(s[-1]))
    #categories = categories.applymap(lambda s: int(s.split('-')[-1]))
    # Drop the original categories column from Dataframe
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop missing values and duplicates from the dataframe
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_colnames, inplace=True)
    
    # Column original seems to be the original (not translated) message in
    # different Languages , we will drop it and use only the english messages
    del df['original']
    
    
    # Columns with only equal values will be dropped
    for col in df.columns:
        if df[col].dtype != 'object':
            if len(pd.unique(df[col]))==1:
                   df = df.drop(col, axis=1)
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - cleaned dataframe
    database_filename(string): the file path to save file .db
    
    OUTPUT
    None
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_messages', engine, index=False, if_exists='replace')
    engine.dispose() 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()