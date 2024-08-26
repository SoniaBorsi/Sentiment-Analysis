import requests
import csv


api_key = 'your API key'
video_id = 'video_id_trump_vs_biden'
video_id_trump_vs_biden = 'qqG96G8YdcE' 
# video_id_trump_shooting = 'YfvSspaJtk'
# video_id_biden_speech = 'Z96N57eV1EA'
url = 'https://www.googleapis.com/youtube/v3/commentThreads'

# Request parameters
params = {
    'part': 'snippet',
    'videoId': video_id,
    'key': api_key,
    'maxResults': 100 
}

def fetch_comments(limit=5000):
    all_comments = []
    next_page_token = None
    
    while len(all_comments) < limit:  
        if next_page_token:
            params['pageToken'] = next_page_token
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()

            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in data.get('items', [])]
            all_comments.extend(comments)

            next_page_token = data.get('nextPageToken')
            
            if not next_page_token:
                break
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break
    
    return all_comments[:limit]  

# fetching 5000 comments
comments = fetch_comments(limit=5000)
