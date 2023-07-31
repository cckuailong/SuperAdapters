import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import json
import argparse
import gradio as gr
from common.db import get_mysql_conn
from common.base import WEB_USERNAME, WEB_PASSWORD


def get_flow():
    payload_uuid = None
    input = None
    output = None
    conn.ping(reconnect=True)
    cur = conn.cursor()
    sql = "select payload_uuid,input,output from playbooks_all where is_check=0"
    cur.execute(sql)
    item = cur.fetchone()
    if item:
        payload_uuid, input, output = item
    else:
        payload_uuid, input, output = None, None, None
    cur.close()

    return payload_uuid, input, output


def passit(uuid):
    conn.ping(reconnect=True)
    cur = conn.cursor()
    try:
        sql = "update playbooks_all set is_check=1 where payload_uuid=%s"
        cur.execute(sql, uuid)
        conn.commit()
    except:
        pass
    cur.close()

    payload_uuid, pb_input, pb_output = get_flow()

    if not payload_uuid:
        payload_uuid = "暂无待标注流量"

    return payload_uuid, pb_input, pb_output

def fixit(uuid, ac_output):
    conn.ping(reconnect=True)
    cur = conn.cursor()
    try:
        sql = "update playbooks_all set is_check=2, output=%s where payload_uuid=%s"
        cur.execute(sql, (ac_output, uuid))
        conn.commit()
    except:
        pass
    cur.close()

    payload_uuid, pb_input, pb_output = get_flow()

    if not payload_uuid:
        payload_uuid = "暂无待标注流量"

    return payload_uuid, pb_input, pb_output

def deleteit(uuid):
    conn.ping(reconnect=True)
    cur = conn.cursor()
    try:
        sql = "update playbooks_all set is_check=-1 where payload_uuid=%s"
        cur.execute(sql, uuid)
        conn.commit()
    except:
        pass
    cur.close()

    payload_uuid, pb_input, pb_output = get_flow()

    if not payload_uuid:
        payload_uuid = "暂无待标注流量"

    return payload_uuid, pb_input, pb_output


parser = argparse.ArgumentParser(description='Label all.')
parser.add_argument('--type', type=str, default='classify', help='classify/chat')
parser.add_argument('--choice', type=str, default='["0", "1"]', help='choices to label')
args, _ = parser.parse_known_args()

conn = get_mysql_conn()
payload_uuid, pb_input, pb_output = get_flow()

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">SuperAdapters -- Label Web</h1>""")

    with gr.Row():
        with gr.Column(scale=1):
            g_payload_uuid = gr.Textbox(label="UUID", value=payload_uuid)
            g_pb_input = gr.TextArea(label="Input", value=pb_input)
        with gr.Column(scale=1):
            if args.type == "classify":
                g_pb_output = gr.Radio(label="Output", choices=json.loads(args.choice), value=pb_output,
                                       interactive=True)
            elif args.type == "chat":
                g_pb_output = gr.Textbox(label="Output", value=pb_output)
            else:
                print("Param type should be 'classify 'or 'chat'")
                sys.exit(-1)
            with gr.Row():
                fixBtn = gr.Button("FixIt!", variant="primary")
                passBtn = gr.Button("Correct! Pass")
            delBtn = gr.Button("Delete the Flow", variant="stop")

    fixBtn.click(fixit, inputs=[g_payload_uuid, g_pb_output], outputs=[g_payload_uuid, g_pb_input, g_pb_output])
    passBtn.click(passit, inputs=[g_payload_uuid], outputs=[g_payload_uuid, g_pb_input, g_pb_output])
    delBtn.click(deleteit, inputs=[g_payload_uuid], outputs=[g_payload_uuid, g_pb_input, g_pb_output])


if os.getenv("DEBUG"):
    demo.queue().launch(share=False, inbrowser=True)
else:
    demo.queue().launch(auth=(WEB_USERNAME, WEB_PASSWORD), server_name="0.0.0.0", server_port=7862, share=False, inbrowser=False)