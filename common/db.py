import os
import pymysql


MYSQL_DB_HOST = os.getenv('LLM_DB_HOST')
MYSQL_DB_PORT = int(os.getenv('LLM_DB_PORT'))
MYSQL_DB_USERNAME = os.getenv('LLM_DB_USERNAME')
MYSQL_DB_PASSWORD = os.getenv('LLM_DB_PASSWORD')
MYSQL_DB_NAME = os.getenv('LLM_DB_NAME')


def get_mysql_conn():
    conn = pymysql.connect(
        host=MYSQL_DB_HOST,
        port=MYSQL_DB_PORT,
        user=MYSQL_DB_USERNAME,
        passwd=MYSQL_DB_PASSWORD,
        db=MYSQL_DB_NAME,
        charset='utf8mb4'
    )

    return conn