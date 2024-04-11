import os
import boto3
from datetime import datetime
import pymysql


def upload_folder_to_s3(local_folder, bucket_name):
    # Create an S3 client using default credentials resolved by boto3
    s3_client = boto3.client('s3')

    # Format the base path with the current date
    today_date = datetime.today().strftime('%Y-%m-%d')
    base_path = f"clients/1/stores/1/{today_date}/img"

    # Walk through the local folder
    for subdir, dirs, files in os.walk(local_folder):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                s3_client.upload_fileobj(
                    data,
                    bucket_name,
                    f"{base_path}/{os.path.relpath(full_path, start=local_folder)}"
                )
    print("Upload complete")


def save_visits(data, store_id, date, host, user, password, database):
    connection = pymysql.connect(host=host, user=user, password=password, database=database)
    try:
        with connection.cursor() as cursor:
            # SQL statement to insert data
            sql = "INSERT INTO visits (`count`, `time`, `store_id`, `date`) VALUES (%s, %s, %s, %s)"
            # Prepare data for insertion
            for item in data:
                cursor.execute(sql, (item['count'], item['time'], store_id, date))
        connection.commit()
    finally:
        connection.close()

if __name__ == '__main__':
    upload_folder_to_s3('/home/diego/Documents/yolov7-tracker/runs/detect/2024_04_03_conce_debug2/imgs_conce_debug', 'videos-mivo')
