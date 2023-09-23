import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('GroceryStoreDataSet.csv', names=['products'], sep=',')
    return df

# Main function
def main():
    st.title('Apriori Association Rule Analysis')

    # Load the data
    df = load_data()

    # Show the first few rows of the dataset
    st.subheader('Dataset')
    st.dataframe(df.head())

    # Perform Apriori analysis
    st.subheader('Apriori Analysis')

    # Define minimum support and confidence values
    min_support = st.slider('Minimum Support', 0.0, 1.0, 0.2)
    min_confidence = st.slider('Minimum Confidence', 0.0, 1.0, 0.6)

    # Perform one-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(df['products'].apply(lambda x: x.split(','))).transform(df['products'].apply(lambda x: x.split(',')))
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Run Apriori
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    st.write(frequent_itemsets)

    # Association rules
    st.subheader('Association Rules')
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    st.write(rules)

if __name__ == '__main__':
    main()













