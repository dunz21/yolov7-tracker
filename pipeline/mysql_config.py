
import pymysql

def get_connection(HOST, ADMIN, PASS, DB):
    connection = pymysql.connect(host=HOST, user=ADMIN, password=PASS, database=DB)
    return connection