import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import functions
import preprocess

st.title('WhatsApp Chat Analyzer')
file = st.file_uploader("Choose a file")

if file:
    df = preprocess.generateDataFrame(file)
    try:
        dayfirst = st.radio(
            "Select Date Format in text file:", ('dd-mm-yy', 'mm-dd-yy'))
        if dayfirst == 'dd-mm-yy':
            dayfirst = True
        else:
            dayfirst = False
        users = functions.getUsers(df)
        users_s = st.sidebar.selectbox("Select User to View Analysis", users)
        selected_user = ""

        if st.sidebar.button("Show Analysis"):
            selected_user = users_s

            st.title("Showing Reults for : " + selected_user)
            df = preprocess.PreProcess(df, dayfirst)
            if selected_user != "Everyone":
                # df of the selected user only
                df = df[df['User'] == selected_user]
            df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count = functions.getStats(
                df)
            st.title("Chat Statistics")
            # stats_c is variable which contains result from getStats funcition
            stats_c = ["Total Messages", "Total Words",
                       "Media Shared", "Links Shared", "Messages Deleted"]
            c1, c2, c3, c4, c5 = st.columns(5)        # 5 columns are made
            with c1:
                st.subheader(stats_c[0])
                st.title(msg_count)
            with c2:
                st.subheader(stats_c[1])
                st.title(word_count)
            with c3:
                st.subheader(stats_c[2])
                st.title(media_cnt)
            with c4:
                st.subheader(stats_c[3])
                st.title(links_cnt)
            with c5:
                st.subheader(stats_c[4])
                st.title(deleted_msgs_cnt)

            # User Activity Count
            if selected_user == 'Everyone':
                # counts the occurence of the each user
                x = df['User'].value_counts().head()
                name = x.index  # name of the user
                count = x.values  # number of message
                st.title("Messaging Frequency")
                st.subheader('Messaging Percentage Count of Users')
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
                        columns={'User': 'name', 'count': 'percent'}))
                with col2:
                    fig, ax = plt.subplots()
                    ax.bar(name, count)
                    ax.set_xlabel("Users")
                    ax.set_ylabel("Message Sent")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # Emoji
            emojiDF = functions.getEmoji(df)
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emojiDF)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emojiDF[1].head(), labels=emojiDF[0].head(),
                       autopct="%0.2f", shadow=True)
                plt.legend()
                st.pyplot(fig)

            # Common Word
            commonWord = functions.MostCommonWords(df)
            fig, ax = plt.subplots()
            ax.bar(commonWord[0], commonWord[1])
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation='vertical')
            st.title('Most Frequent Words Used In Chat')
            st.pyplot(fig)

            # Monthly Timeline
            # gets the timeline which contains year month and message count
            timeline = functions.getMonthlyTimeline(df)
            # this returns the figure as well as the axes objects
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Message'])
            ax.set_xlabel("Month")
            ax.set_ylabel("Messages Sent")
            plt.xticks(rotation='vertical')
            st.title('Monthly Timeline')
            st.pyplot(fig)

            # Daily Timeline
            functions.dailytimeline(df)

            st.title('Most Busy Days')
            functions.WeekAct(df)
            st.title('Most Busy Months')
            functions.MonthAct(df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = functions.create_wordcloud(df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = functions.activity_heatmap(df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # sentiment analysis
            messages = df['Message'].tolist()

            sentiment_scores = preprocess.preprocess_and_analyze_sentiment(
                messages)
            average_score, positive_messages, negative_messages, neutral_messages = preprocess.analyze_sentiment_scores(
                sentiment_scores, messages)
            # Display sentiment analysis results
            st.title('Sentiment Analysis of WhatsApp Chat')
            st.write('**Average Sentiment Score:**', average_score)
            st.write('**Number of Positive Messages:**',
                     len(positive_messages))
            st.write('**Number of Negative Messages:**',
                     len(negative_messages))
            st.write('**Number of Neutral Messages:**', len(neutral_messages))

    except Exception as e:
        st.subheader("Unable to Process Your Request")
