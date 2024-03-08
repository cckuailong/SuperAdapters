import json
import hashlib
from common.db import conn


def check_token(user, token, data):
    conn.ping(reconnect=True)
    cur = conn.cursor()
    sql = "select `key` from api_user where user=%s"
    cur.execute(sql, user)
    item = cur.fetchone()
    cur.close()
    if not item:
        return False
    key = item[0]

    hash_md5 = hashlib.md5()
    hash_md5.update((json.dumps(data) + key).encode())
    md5_str = hash_md5.hexdigest()

    if token.strip() == md5_str:
        return True
    else:
        return False
