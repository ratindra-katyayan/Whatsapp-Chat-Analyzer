from collections import Counter
import pandas as pd
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import urlextract
import emoji
from wordcloud import WordCloud


def getUsers(df):
    users = df['User'].unique().tolist()
    users.sort()
    users.remove('Notifications')
    users.insert(0, 'Everyone')
    return users


def getStats(df):
    # checks if any message contains media
    media = df[df['Message'] == "<Media omitted> "]
    media_cnt = media.shape[0]
    df.drop(media.index, inplace=True)
    # checks deleted messages
    deleted_msgs = df[df['Message'] == "This message was deleted "]
    deleted_msgs_cnt = deleted_msgs.shape[0]
    df.drop(deleted_msgs.index, inplace=True)
    temp = df[df['User'] == 'Notifications']  # checks group notifications
    df.drop(temp.index, inplace=True)
    extractor = urlextract.URLExtract()
    links = []
    for msg in df['Message']:
        x = extractor.find_urls(msg)  # checks urls or links
        if x:
            links.extend(x)
    links_cnt = len(links)
    word_list = []
    for msg in df['Message']:
        word_list.extend(msg.split())  # splits the words with respect to the
    word_count = len(word_list)
    msg_count = df.shape[0]
    return df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count


def getEmoji(df):
    emojis = []
    for message in df['Message']:
        # checks for emojis in the messages using unicode
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))


def getMonthlyTimeline(df):

    df.columns = df.columns.str.strip()
    df = df.reset_index()
    # first group by year and then month and takes no. of messages
    timeline = df.groupby(['year', 'month']).count()['Message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        # time list adds month and year along with the messages
        time.append(str(timeline['month'][i]) + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


def MostCommonWords(df):
    f = open('stop_hinglish.txt')
    stop_words = f.read()
    f.close()
    words = []
    for message in df['Message']:
        for word in message.lower().split():
            if word not in stop_words:  # checks whether it is there in the stop words or not
                words.append(word)
    # ountts frequency and the words
    return pd.DataFrame(Counter(words).most_common(20))


def dailytimeline(df):
    df['taarek'] = df['Date']
    daily_timeline = df.groupby('taarek').count(
    )['Message'].reset_index()  # group by the date column
    fig, ax = plt.subplots()
    # ax.figure(figsize=(100, 80))
    ax.plot(daily_timeline['taarek'], daily_timeline['Message'])
    ax.set_ylabel("Messages Sent")
    st.title('Daily Timeline')
    st.pyplot(fig)


def WeekAct(df):
    # counts how many times each day occurs individually  i.e. numbers of messages in each day
    x = df['day'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(x.index, x.values)
    ax.set_xlabel("Days")
    ax.set_ylabel("Message Sent")
    plt.xticks(rotation='vertical')
    st.pyplot(fig)


def MonthAct(df):
    # counts how many times each month occurs individually  i.e. numbers of messages in each month
    x = df['month_name'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(x.index, x.values)
    ax.set_xlabel("Months")
    ax.set_ylabel("Message Sent")
    plt.xticks(rotation='vertical')
    st.pyplot(fig)


def activity_heatmap(df):
    period = []
    for hour in df[['day', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))  # 23-00
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))  # 00-01
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    # it counts the occurrences of messages for each day-period combination and fill any missing value with 0
    user_heatmap = df.pivot_table(
        index='day', columns='period', values='Message', aggfunc='count').fillna(0)
    return user_heatmap


def create_wordcloud(df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()  # words not to be displayed
    f.close()

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    # background white image of word cloud
    wc = WordCloud(width=500, height=500, min_font_size=10,
                   background_color='white')
    # check for the non-display words
    df['Message'] = df['Message'].apply(remove_stop_words)
    df_wc = wc.generate(df['Message'].str.cat(sep=" "))  # creates word cloud
    return df_wc


def analyze_sentiment_trends(sentiment_scores,messages):

    # Calculate average sentiment score
    average_score = sum(sentiment_scores) / len(sentiment_scores)

    # Categorize messages based on sentiment
    positive_messages = [message for message, score in zip(
        messages, sentiment_scores) if score > 0]
    negative_messages = [message for message, score in zip(
        messages, sentiment_scores) if score < 0]
    neutral_messages = [message for message, score in zip(
        messages, sentiment_scores) if score == 0]

    return average_score, positive_messages, negative_messages, neutral_messages
