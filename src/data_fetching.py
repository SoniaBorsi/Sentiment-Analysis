import requests
import csv


api_key = 'your API key'
video_id = 'video ID'
url = 'URL video'

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

comments = fetch_comments(limit=5000)

# # Save comments to a CSV file
# with open('comments.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Comment Number', 'Comment Content'])  # Header row
#     for idx, comment in enumerate(comments, start=1):
#         writer.writerow([idx, comment])

# print(f"Saved {len(comments)} comments to comments.csv.")
