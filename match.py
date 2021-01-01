import xml.etree.ElementTree as ET
from collections import defaultdict


# Topics in dictionary
topics = ET.parse('topics-rnd5.xml')
root = topics.getroot()
even_topics = [{'topic_id': int(topic.attrib['number']), 
                'query': topic[0].text, 
                'question': topic[1].text, 
                'narrative': topic[2].text} for topic in root if int(topic.attrib['number']) % 2 == 0]

odd_topics = [{'topic_id': int(topic.attrib['number']), 
               'query': topic[0].text, 
               'question': topic[1].text, 
               'narrative': topic[2].text} for topic in root  if int(topic.attrib['number']) % 2 == 1]


# Only Docs with Relevancy Judgements
topic_with_relevancy = defaultdict(list)

with open("relevancy-judgements.txt", "r") as f:
    text = f.read()
    text = text.split('\n')
    samp = ""
    for line in text:
        items = line.split()

        topic_id = int(items[0])
        cord_uid = items[2]
        relevancy = int(items[3])

        topic_with_relevancy[cord_uid].append({
            'topic_id' : topic_id,
            'relevancy': relevancy
        })



