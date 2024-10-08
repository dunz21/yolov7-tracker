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
    def queue_videos(cls, channel_id=None):
        url = f"{cls.get_base_url()}/api/queue-videos"
        params = {}
        
        # If channel_id is provided, add it to the request parameters
        if channel_id is not None:
            params['channel_id'] = channel_id
        
        # Make the GET request with optional parameters
        response = requests.get(url, params=params).json()
        return response
    
    @classmethod
    def get_zones(cls):
        url = f"{cls.get_base_url()}/api/zones"
        response = requests.get(url).json()
        return response
    @classmethod
    
    def get_inference_params(cls):
        url = f"{cls.get_base_url()}/api/inference-params"
        response = requests.get(url).json()
        return response
    
    @classmethod
    def get_finished_queue_videos(cls, machine_name):
        url = f"{cls.get_base_url()}/api/queue-videos/finished-queue"
        params = {'machine_name': machine_name}
        headers = {'Accept': 'application/json'}  # Specify that the client expects a JSON response

        response = requests.get(url, params=params, headers=headers).json()
        return response

    @classmethod
    def update_video_status(cls, video_id, status, machine_name=None):
        url = f"{cls.get_base_url()}/api/queue-videos/{video_id}/status"
        data = {
            'status': status,
            'machine_name': machine_name  # Add machine_name to the data payload
        }
        # Filter out None values if machine_name is not provided
        data = {key: value for key, value in data.items() if value is not None}
        headers = {'Content-Type': 'application/json'}  # Ensure headers specify JSON
        response = requests.put(url, json=data, headers=headers)  # Use json instead of data for correct content type
        return response
    
    @classmethod
    def update_video_process_status(cls, video_id, progress,status):
        url = f"{cls.get_base_url()}/api/queue-videos/{video_id}/progress"
        response = requests.put(url, data={'progress': progress, 'status': status})
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
    def post_queue_video_result(cls, queue_video_id, time_start=None, time_end=None, timings=None, total_frames=None, total_duration=None, fps=None, metadata=None, error=None, results=None):
        url = f"{cls.get_base_url()}/api/queue-video-results"
        headers = {'Content-Type': 'application/json'}
        data = {
            'queue_video_id': queue_video_id,
            'time_start': time_start,
            'time_end': time_end,
            'timings': timings,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'fps': fps,
            'metadata': metadata,
            'error': error,
            'results': results
        }
        data = {k: v for k, v in data.items() if v is not None}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Successfully posted queue video result with ID {queue_video_id}")
        else:
            print(f"Failed to post queue video result. Status code: {response.status_code}, Response: {response.text}")
