import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json = {
        "model" : "bge-m3",
        "input" : text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json = {
        # "model" : "deepseek-r1",
        "model" : "llama3.2",
        "Prompt" : prompt,
        "stream" : False
    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question?: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding'].values),[question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df['Title','Text'])

Prompt = f''' Raj shamani is teaching through podcast.
Here are video chunks containing video title, chunk ID ,start time in second,end time in seconds, the text at that time: 

{new_df[['Title','Start','End','chunk_id','Text']].to_json(orient = "records")}
----------------------------------------
"{incoming_query}"
user asked this question related to the video chunks, you have to answer in human way(don't mention the above format , its just for you) where and how much content is taught where (in which video and at what timestamp) and guide the user to go to that particular video.
if user asks unrelated question, tell him that you can only answer questions related to the podcast
'''

with open("Prompt.txt","w") as f:
    f.write(Prompt)

response = inference(Prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)
# for index,item in new_df.iterrows():
#     print(index,item["Title"],item["chunk_id"],item["Text"],item["Start"],item["End"])