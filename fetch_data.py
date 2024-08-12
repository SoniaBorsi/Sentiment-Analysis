import requests
import csv

# Your API key
api_key = 'AIzaSyAwHjR8O6X0Q3wEpzoiWgvT5c5PyoauNMo'
# Video ID from the URL
video_id = 'qqG96G8YdcE'
# YouTube API URL
url = 'https://www.googleapis.com/youtube/v3/commentThreads'

# Request parameters
params = {
    'part': 'snippet',
    'videoId': video_id,
    'key': api_key,
    'maxResults': 100  # Fetch up to 100 comments per request
}

def fetch_comments():
    all_comments = []
    next_page_token = None
    
    while len(all_comments) < 10000:
        if next_page_token:
            params['pageToken'] = next_page_token
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Extract comments from the response
            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in data.get('items', [])]
            all_comments.extend(comments)
            
            # Check if there's a next page
            next_page_token = data.get('nextPageToken')
            
            if not next_page_token:
                break
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break
    
    return all_comments[:1000]  # Return only the first 1000 comments

# Fetch comments
comments = fetch_comments()

# Save comments to a CSV file
with open('comments.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Comment Number', 'Comment Content'])  # Header row
    for idx, comment in enumerate(comments, start=1):
        writer.writerow([idx, comment])

print(f"Saved {len(comments)} comments to comments.csv.")
