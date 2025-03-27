import os
import json
import pymysql

MYSQL_DB_HOST = os.getenv('LLM_DB_HOST')
MYSQL_DB_PORT = int(os.getenv('LLM_DB_PORT', 3306))
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


conn = get_mysql_conn()
cur = conn.cursor()
sql = "select instruction,input,output from playbooks_all where iteration=%s and `type`='train' and is_check > 0"
cur.execute(sql, "guardlm_2025_0320")
items = cur.fetchall()
cur.close()
conn.close()

train_data_set = []

for item in items:
    instruction, input, output = item
    train_data_set.append({
        "instruction": instruction,
        "input": input,
        "output": output
    })

with open("/root/LLaMA-Factory/data/guardlm_train.json", "w") as f:
    f.write(json.dumps(train_data_set))
