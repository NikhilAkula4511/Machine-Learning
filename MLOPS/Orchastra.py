from prefect import task, flow
import pandas as pd
from sklearn.model_selection import train_test_split
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)
@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def Preprocess(X_train,X_test) :
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocab2 = TfidfVectorizer()
    X_train_tfidf2 = vocab2.fit_transform(X_train)
    X_test_tfidf2 = vocab2.transform(X_test)

    return X_train_tfidf2,X_test_tfidf2

@task
def Model(X_train_trans,y_train,X_test_trans,y_test):
    from sklearn.metrics import accuracy_score,classification_report
    from sklearn.naive_bayes import MultinomialNB
    Mnb = MultinomialNB()
    Mnb.fit(X_train_trans,y_train)


    y_pred = Mnb.predict(X_test_trans)
    accuracy = accuracy_score(y_test,y_pred)
    cm = classification_report(y_test,y_pred)

    return y_pred,accuracy,cm

    
    

# Workflow

@flow(name="Sentiment Analysis")
def workflow():
    DATA_PATH = r'C:\\Users\\Hello\\INTERNSHIP\\reviews_badminton\data.csv'
    # Load data
    df = load_data(DATA_PATH)

    INPUTS = 'Review text'
    OUTPUT = 'Ratings'
    # Identify Inputs and Output
    X, y = split_inputs_output(df, INPUTS, OUTPUT)
    y = y.apply(lambda x: 'Negative' if x < 3 else ('Neutral' if x == 3 else 'Positive'))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    def clean(doc): # doc is a string of text

        doc = str(doc)

        doc = doc.replace("READ MORE", "")
        
        # Remove punctuation and numbers.
        doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

        # Converting to lower case
        doc = doc.lower()
        
        # Tokenization
        tokens = nltk.word_tokenize(doc)

        # Lemmatize
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Stop word removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
        
        # Join and return
        return " ".join(filtered_tokens)
    X_train_clean = X_train.apply(lambda text: clean(text))
    X_test_clean = X_test.apply(lambda text: clean(text))

    X_train_trans,x_test_trans = Preprocess(X_train_clean,X_test_clean)

    prediction,accuracy,confusion_matr = Model(X_train_trans,y_train,x_test_trans,y_test)

    return prediction,accuracy,confusion_matr

    




if __name__ == "__main__":
    workflow.serve(
        name="sentiment_analysis_project",
        cron="* * * * *"
    )