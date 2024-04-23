import datetime


def seconds_to_time(seconds):
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)
    # Add the timedelta to a minimal datetime object
    time = (datetime.datetime.min + td).time()
    # Convert to a string format
    return time.strftime("%H:%M:%S")