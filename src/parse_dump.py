import lxml.etree as ET
import pandas as pd
from bs4 import BeautifulSoup
import os

def clean_html(text):
    if not text:
        return ""
    return BeautifulSoup(text, 'html.parser').get_text()

def parse_posts(xml_path):
    questions = []
    answers = []
    
    for event, elem in ET.iterparse(xml_path, events=('end',), tag='row'):
        post_type = elem.get('PostTypeId')
        
        if post_type == '1':  # question
            questions.append({
                'Id': int(elem.get('Id')),
                'Title': elem.get('Title', ''),
                'Body': clean_html(elem.get('Body', '')),
                'Tags': elem.get('Tags', ''),
                'Score': int(elem.get('Score', 0)),
                'AcceptedAnswerId': elem.get('AcceptedAnswerId'),
                'AnswerCount': int(elem.get('AnswerCount', 0)),
                'CreationDate': elem.get('CreationDate'),
            })
        elif post_type == '2':  # answer
            answers.append({
                'Id': int(elem.get('Id')),
                'ParentId': int(elem.get('ParentId')),
                'Body': clean_html(elem.get('Body', '')),
                'Score': int(elem.get('Score', 0)),
            })
        
        elem.clear()
    
    return pd.DataFrame(questions), pd.DataFrame(answers)

def parse_postlinks(xml_path):
    pairs = []
    for event, elem in ET.iterparse(xml_path, events=('end',), tag='row'):
        pairs.append({
            'PostId': int(elem.get('PostId')),
            'RelatedPostId': int(elem.get('RelatedPostId')),
            'LinkTypeId': int(elem.get('LinkTypeId')),
        })
        elem.clear()
    return pd.DataFrame(pairs)

if __name__ == '__main__':
    raw = 'data/raw/'
    processed = 'data/processed/'
    os.makedirs(processed, exist_ok=True)
    
    print("Parsing Posts.xml...")
    questions, answers = parse_posts(raw + 'Posts.xml')
    questions.to_csv(processed + 'questions.csv', index=False)
    answers.to_csv(processed + 'answers.csv', index=False)
    print(f"Questions: {len(questions)}, Answers: {len(answers)}")
    
    print("Parsing PostLinks.xml...")
    pairs = parse_postlinks(raw + 'PostLinks.xml')
    pairs.to_csv(processed + 'pairs.csv', index=False)
    print(f"Pairs: {len(pairs)}")