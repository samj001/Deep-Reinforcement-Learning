import sqlite3

from elasticsearch import Elasticsearch

def getTopics():
    sqlite_file = ".\\trec-dd-jig\\jig\\truth.db" #address of the topic database
    topic_table = "topic"
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    #c.execute('SELECT * FROM {tt}'.format(tt = topic_table))
    #all_rows = c.fetchall()
    c.execute('SELECT topic_name FROM {tt}'.format(tt = topic_table))
    all_topics = c.fetchall()
    print(all_topics)
    conn.commit()
    conn.close()
    return all_topics

def main():
    topic = getTopics()
    print(topic)

if __name__ == "__main__":
    main()