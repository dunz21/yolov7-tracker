import requests

class APIConfig:
    _instance = None
    base_url = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(APIConfig, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def initialize(cls, base_url):
        cls.base_url = base_url

    @classmethod
    def get_base_url(cls):
        if not cls.base_url:
            raise ValueError("Base URL is not set. Please initialize the APIConfig with a base URL.")
        return cls.base_url

    @classmethod
    def queue_videos(cls):
        url = f"{cls.get_base_url()}/api/queue-videos"
        response = requests.get(url).json()
        return response

    @classmethod
    def update_video_status(cls, video_id, status):
        url = f"{cls.get_base_url()}/api/queue-videos/{video_id}/status"
        response = requests.put(url, data={'status': status})
        return response

    @classmethod
    def save_visits_per_hour(cls, list_visits_group_by_hour, store_id, date, visit_type_id=''):
        url = f"{cls.get_base_url()}/api/save-visits-per-hour"
        headers = {'Content-Type': 'application/json'}
        
        data = []
        for item in list_visits_group_by_hour:
            data.append({
                'count': item['count'],
                'time': item['time_calculated'],
                'store_id': store_id,
                'date': date,
                'visit_type_id': visit_type_id,
            })
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Inserted {len(list_visits_group_by_hour)} visits successfully")
        else:
            print(f"Failed to insert visits. Status code: {response.status_code}, Response: {response.text}")


    @classmethod
    def get_unique_sales(cls, store_id, date):
        url = f"{cls.get_base_url()}/api/unique-sales"
        headers = {'Content-Type': 'application/json'}
        params = {
            'store_id': store_id,
            'date': date
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            unique_sales = response.json().get('unique_sales')
            print(f"Unique sales for store {store_id} on {date}: {unique_sales}")
            return unique_sales
        else:
            print(f"Failed to retrieve unique sales. Status code: {response.status_code}, Response: {response.text}")
            return 0
    
    @classmethod
    def save_event_timestamps(cls, list_event_timestamps):
        url = f"{cls.get_base_url()}/api/save-event-timestamps"
        headers = {'Content-Type': 'application/json'}
        
        # Data should be a list of dictionaries
        response = requests.post(url, headers=headers, json=list_event_timestamps)
        
        if response.status_code == 201:
            print("Data inserted successfully")
        else:
            print(f"Failed to insert data. Status code: {response.status_code}, Response: {response.text}")
    
    @classmethod
    def save_short_visits(cls, short_video_clips_urls, date, store_id):
        url = f"{cls.get_base_url()}/api/save-short-visits"
        headers = {'Content-Type': 'application/json'}
        
        data = []
        for item in short_video_clips_urls:
            data.append({
                'url': item['url'],
                'date': date,
                'store_id': store_id,
            })
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Inserted {len(short_video_clips_urls)} short visits successfully")
        else:
            print(f"Failed to insert short visits. Status code: {response.status_code}, Response: {response.text}")

                
    @classmethod
    def save_sankey_diagram(cls,store_id, date, exterior=None, interior=None, pos=None, short_visit=None):
        url = f"{cls.get_base_url()}/api/sankey"
        headers = {'Content-Type': 'application/json'}
        data = {
            'store_id': store_id,
            'date': date,
            'exterior': exterior,
            'interior': interior,
            'pos': pos,
            'short_visit': short_visit,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Inserted sankey diagram for store {store_id} on date {date}")
        else:
            print(f"Failed to insert sankey diagram. Status code: {response.status_code}, Response: {response.text}")
            
    @classmethod
    def save_reid_matches(cls,store_id, date, reid_matches=[]):
        url = f"{cls.get_base_url()}/api/reid-matches"
        headers = {'Content-Type': 'application/json'}
        data = {
            'store_id': store_id,
            'date': date,
            'reid_matches': reid_matches,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Inserted reid matches for store {store_id} on date {date}")
        else:
            print(f"Failed to insert reid matches. Status code: {response.status_code}, Response: {response.text}")
                
    @classmethod
    def post_queue_video_result(cls, queue_video_id, model_name, results):
        url = f"{cls.get_base_url()}/api/queue-video-results"
        headers = {'Content-Type': 'application/json'}
        data = {
            'queue_video_id': queue_video_id,
            'model_name': model_name,
            'results': results
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Successfully posted queue video result with ID {queue_video_id}")
        else:
            print(f"Failed to post queue video result. Status code: {response.status_code}, Response: {response.text}")