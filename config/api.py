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
    def save_visits_per_hour(cls, list_visits_group_by_hour, store_id, date):
        url = f"{cls.get_base_url()}/api/save-visits-per-hour"
        headers = {'Content-Type': 'application/json'}
        for item in list_visits_group_by_hour:
            data = {
                'count': item['count'],
                'time': item['time_calculated'],
                'store_id': store_id,
                'date': date,
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                print(f"Inserted {item['count']} visits at {item['time_calculated']}")
            else:
                print(f"Failed to insert {item['count']} visits at {item['time_calculated']}. Status code: {response.status_code}, Response: {response.text}")

    @classmethod
    def save_short_visits(cls, short_video_clips_urls, date, store_id):
        url = f"{cls.get_base_url()}/api/save-short-visits"
        headers = {'Content-Type': 'application/json'}
        for item in short_video_clips_urls:
            data = {
                'url': item['url'],
                'date': date,
                'store_id': store_id,
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                print(f"Inserted short visit for URL {item['url']}")
            else:
                print(f"Failed to insert short visit for URL {item['url']}. Status code: {response.status_code}, Response: {response.text}")