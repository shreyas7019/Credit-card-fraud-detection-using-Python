import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import timeit

# Replace 'YOUR_POWER_BI_REPORT_LINK' with the actual link to your hosted Power BI report
POWER_BI_REPORT_LINK = 'https://app.powerbi.com/reportEmbed?reportId=3efcf72f-87ba-4255-b1b5-1b3c513a7b4a&autoAuth=true&ctid=355dba84-7f18-4b49-9305-816dd2d6864b'

@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

def show_homepage():
    st.title('Credit Card Fraud Detection')
    st.write("Welcome to the Credit Card Fraud Detection app!")
    st.write("This app helps you explore a credit card fraud dataset and run an Extra Trees classifier for fraud detection.")

def show_dataset_info(df):
    st.header('Dataset Overview')
    if st.checkbox('Show Sample Data'):
        st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data description: \n', df.describe())

def show_fraud_valid_details(df):
    st.header('Fraud and Valid Transaction Details')
    fraud = df[df.Class == 1]
    valid = df[df.Class == 0]
    outlier_percentage = (df.Class.value_counts()[1] / df.Class.value_counts()[0]) * 100
    st.write('Fraudulent transactions are: %.3f%%' % outlier_percentage)
    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))

def perform_train_test_split(df):
    st.header('Train-Test Split')
    size = st.slider('Test Set Size', min_value=0.2, max_value=0.4)
    X = df.drop(['Class'], axis=1)
    y = df.Class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    if st.checkbox('Show the shape of training and test set features and labels'):
        st.write('X_train: ', X_train.shape)
        st.write('y_train: ', y_train.shape)
        st.write('X_test: ', X_test.shape)
        st.write('y_test: ', y_test.shape)

    return X_train, X_test, y_train, y_test

def show_feature_importance(X_train, y_train):
    st.header('Feature Importance')
    model = ExtraTreesClassifier(random_state=42)
    importance = model.fit(X_train, y_train).feature_importances_

    if st.checkbox('Show plot of feature importance'):
        feature_importance_fig, feature_importance_ax = plt.subplots()
        feature_importance_ax.bar([x for x in range(len(importance))], importance)
        feature_importance_ax.set_title('Feature Importance')
        feature_importance_ax.set_xlabel('Feature (Variable Number)')
        feature_importance_ax.set_ylabel('Importance')
        st.pyplot(feature_importance_fig)

def compute_performance(model, X_train, y_train, X_test, y_test):
    st.subheader('Model Performance Metrics')
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    st.write('Accuracy:', scores)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write('Confusion Matrix:', cm)

    cr = classification_report(y_test, y_pred)
    st.write('Classification Report:', cr)

    mcc = matthews_corrcoef(y_test, y_pred)
    st.write('Matthews Correlation Coefficient:', mcc)

    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for performance computation: %.2f minutes' % (elapsed / 60))

def show_graphs(df):
    st.header('Visualizing Different Types of Graphs')
    graph_type = st.selectbox('Select Graph Type', ['Count Plot', 'Box Plot', 'Violin Plot', 'Histogram', 'Scatter Plot', 'Bubble Plot', 'Correlation Heatmap'])

    if graph_type == 'Count Plot':
        st.subheader('Count Plot of Fraudulent and Valid Transactions')
        count_plot_fig, count_plot_ax = plt.subplots()
        sns.countplot(x='Class', data=df, ax=count_plot_ax)
        st.pyplot(count_plot_fig)

    elif graph_type == 'Box Plot':
        st.subheader('Box Plot of Amount by Class')
        box_plot_fig, box_plot_ax = plt.subplots()
        sns.boxplot(x='Class', y='Amount', data=df, ax=box_plot_ax)
        st.pyplot(box_plot_fig)

    elif graph_type == 'Violin Plot':
        st.subheader('Violin Plot of Amount by Class')
        violin_plot_fig, violin_plot_ax = plt.subplots()
        sns.violinplot(x='Class', y='Amount', data=df, ax=violin_plot_ax)
        st.pyplot(violin_plot_fig)

    elif graph_type == 'Histogram':
        st.subheader('Histogram of Transaction Amount')
        hist_fig, hist_ax = plt.subplots()
        hist_ax.hist(df['Amount'], bins=50, color='blue', alpha=0.7)
        hist_ax.set_title('Histogram of Transaction Amount')
        hist_ax.set_xlabel('Transaction Amount')
        hist_ax.set_ylabel('Frequency')
        st.pyplot(hist_fig)

    elif graph_type == 'Scatter Plot':
        st.subheader('Scatter Plot of Time vs Amount')
        scatter_plot_fig, scatter_plot_ax = plt.subplots()
        scatter_plot_ax.scatter(df['Time'], df['Amount'], alpha=0.5)
        scatter_plot_ax.set_title('Scatter Plot of Time vs Amount')
        scatter_plot_ax.set_xlabel('Time')
        scatter_plot_ax.set_ylabel('Amount')
        st.pyplot(scatter_plot_fig)

    elif graph_type == 'Bubble Plot':
        st.subheader('Bubble Plot of Time vs Amount with Class')
        bubble_plot_fig, bubble_plot_ax = plt.subplots()
        bubble_plot_ax.scatter(df['Time'], df['Amount'], s=df['Class'] * 10, alpha=0.5)
        bubble_plot_ax.set_title('Bubble Plot of Time vs Amount with Class')
        bubble_plot_ax.set_xlabel('Time')
        bubble_plot_ax.set_ylabel('Amount')
        st.pyplot(bubble_plot_fig)

    elif graph_type == 'Correlation Heatmap':
        st.subheader('Correlation Heatmap of Features')
        correlation_matrix = df.corr()
        heatmap_fig, heatmap_ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5, ax=heatmap_ax)
        st.pyplot(heatmap_fig)

def show_power_bi_report():
    st.header('Power BI Report')
    report_width = 800
    report_height = 600

    report_embed_code = f'<iframe width="{report_width}" height="{report_height}" src="{POWER_BI_REPORT_LINK}" frameborder="0" allowFullScreen="true" style="border: 1px solid #dddddd;"></iframe>'
    st.markdown(report_embed_code, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title='Credit Card Fraud Detection', page_icon=':credit_card:')
    df = load_data()

    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None

    page = st.sidebar.radio('Navigation', ['Home', 'Dataset Overview', 'Transaction Details', 'Train-Test Split', 'Feature Importance', 'Graphs', 'Power BI Report'])

    if page == 'Home':
        show_homepage()

    elif page == 'Dataset Overview':
        show_dataset_info(df)

    elif page == 'Transaction Details':
        show_fraud_valid_details(df)

    elif page == 'Train-Test Split':
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = perform_train_test_split(df)

    elif page == 'Feature Importance':
        if st.session_state.X_train is not None:
            show_feature_importance(st.session_state.X_train, st.session_state.y_train)
        else:
            st.warning('Please perform Train-Test Split first to compute feature importance.')

    elif page == 'Graphs':
        show_graphs(df)

    elif page == 'Power BI Report':
        show_power_bi_report()

    st.sidebar.title('Credit Card Fraud Detection')
    if st.sidebar.checkbox('Run a credit card fraud detection model'):
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)

        if st.sidebar.checkbox('Run the model'):
            if st.session_state.X_train is not None:
                compute_performance(model, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
            else:
                st.warning('Please perform Train-Test Split first to run the model.')

if __name__ == '__main__':
    main()
bg_img = '''
<style>
        [data-testid="stAppViewContainer"] {
        background-image: url('https://wallpapers.com/images/high/mastercard-and-visa-credit-cards-macro-shot-256cbhdypwr7l9r5.webp');
        background-size: cover;
        background-repeat: no-repeat;
        }
</style>
'''
st.markdown(bg_img, unsafe_allow_html=True)
nav_bar_bg = '''
    <style>
        [data-testid="stSidebar"] {
            background-image: url('https://wallpapers.com/images/high/visa-credit-card-black-concept-design-t9xemhycu09sne7l.webp');
            background-size: cover;
            background-repeat: no-repeat;
            color: white !important;  /* Set font color to white */
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            color: white !important;  /* Set font color to white when expanded */
        }
    </style>
    '''
st.markdown(nav_bar_bg, unsafe_allow_html=True)
