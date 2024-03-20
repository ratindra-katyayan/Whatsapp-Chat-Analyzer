import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def generateDataFrame(file):
    data = file.read().decode("utf-8")
    data = data.replace('\u202f', ' ')
    data = data.replace('\n', ' ')
    dt_format = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM\s|PM\s|am\s|pm\s)?-\s'
    msgs = re.split(dt_format, data)[1:]
    date_times = re.findall(dt_format, data)
    date = []
    time = []
    for dt in date_times:
        date.append(re.search('\d{1,2}/\d{1,2}/\d{2,4}', dt).group())
        time.append(re.search('\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?', dt).group())
    users = []
    message = []
    for m in msgs:
        s = re.split('([\w\W]+?):\s', m)
        if (len(s) < 3):
            users.append("Notifications")
            message.append(s[0])
        else:
            users.append(s[1])
            message.append(s[2])
    df = pd.DataFrame(list(zip(date, time, users, message)),
                      columns=["Date", "Time(U)", "User", "Message"])
    return df


def PreProcess(df, dayf):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=dayf)
    df['Time'] = pd.to_datetime(df['Time(U)']).dt.time
    df['year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['date'] = df['Date'].apply(lambda x: int(str(x)[8:10]))
    df['day'] = df['Date'].apply(lambda x: x.day_name())
    df['hour'] = df['Time'].apply(lambda x: int(str(x)[:2]))
    df['month_name'] = df['Date'].apply(lambda x: x.month_name())
    return df


nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_message(message):
    # Tokenize the message
    tokens = word_tokenize(message.lower())

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    preprocessed_message = ' '.join(tokens)

    return preprocessed_message


def analyze_sentiment(message):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(message)
    # compound score ranges from -1 (negative) to 1 (positive)
    return sentiment_scores['compound']


def preprocess_and_analyze_sentiment(messages):
    # preprocessed_messages = preprocess_message(messages)

    sentiment_scores = []

    for message in messages:
        preprocessed_message = preprocess_message(message)
        sentiment_score = analyze_sentiment(preprocessed_message)
        sentiment_scores.append(sentiment_score)

    return sentiment_scores

# Update the analyze_sentiment function to categorize messages


def analyze_sentiment_scores(sentiment_scores, messages):
    positive_messages = []
    negative_messages = []
    neutral_messages = []

    for score, message in zip(sentiment_scores, messages):
        if score > 0:
            positive_messages.append(message)
        elif score < 0:
            negative_messages.append(message)
        else:
            neutral_messages.append(message)

    # Calculate average sentiment score
    average_score = sum(sentiment_scores) / len(sentiment_scores)

    return average_score, positive_messages, negative_messages, neutral_messages
