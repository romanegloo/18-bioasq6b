#!/usr/bin/env python3
"""galago building index process stalls frequently due to unstable handling
of distributed indexing.

Update. indexing in 'drmaa' mode is now available on kyric server. Submitting
the galago-build command via qsub will achieve much faster performance.
"""



import os
import signal
import subprocess
import time
from lxml import html
import requests
from twilio.rest import Client

server = "http://10.33.21.155:45678"
job_dir = "/home/jno236/data/tmp/galago.job"
interim = 3 * 60
# build index parameters
params = [
    '--server=true',
    '--port=45678',
    '--fileType=trectext',
    '--inputPath=/home/jno236/data/medline17_trectext',
    '--indexPath=/home/jno236/data/galago-medline-full-jan09.idx',
    '--galagoJobDir=/home/jno236/data/tmp/galago.job',
    '--deleteJobDir=true',
    '--fieldIndex=true',
    '--tokenizer/class=org.lemurproject.galago.core.parse.TagTokenizer',
    '--tokenizer/fields+docno',
    '--tokenizer/fields+title',
    '--tokenizer/fields+journal_title',
    '--tokenizer/fields+medline_ta',
    '--tokenizer/fields+chemical_ui',
    '--tokenizer/fields+chemical',
    '--tokenizer/fields+mesh_ui',
    '--tokenizer/fields+mesh_desc',
    '--tokenizer/fields+mesh_qual',
    '--tokenizer/fields+keywords',
    '--mode=local',
    '--distrib=8096'
]
command = ['galago', 'build']
command.extend(params)


# notifier
def sms(msg):
    num_from = "+17655166130"
    num_to = "+18123457891"
    account_sid = 'AC339ad853d7e4105465ff50d3d5be03eb'
    auth_token = '81b060bae2907aa16c7d8ff92cbca48e'

    print('sending message: {}'.format(msg))
    client = Client(account_sid, auth_token)
    resp = client.messages.create(to=num_to, from_=num_from, body=msg)
    return resp

# RUN~!
# check the job directory; If it exists, continuing the process, else confirm
#  that it starts the process from the first.
if not os.path.exists(job_dir):
    confirm = input('Do you confirm running galago build from the first?'
                    ' [yes/No]  ')
    if not confirm.lower().startswith('y'):
        exit(1)

process = None
prev_status = [None] * 4
completed = 0
notify_completed = 0  # at hundred-th
while True:  # re-run the process if necessary
    if process is None:
        process = subprocess.Popen(command)
        time.sleep(120)  # wait for server running up
    else:
        if prev_status[0] == 'parsePostings':
            time.sleep(interim)
        else:
            time.sleep(20 * interim)

    # check server status
    print("Monitoring...")
    server_msg = requests.get(server)
    if server_msg.status_code != 200:
        confirm = input('Server is not running, do you want to continue '
                        'building index? [yes/No]')
        if not confirm.lower().startswith('y'):
            exit(1)

    tree = html.fromstring(server_msg.content)
    status = [len(tree.xpath("//tr[@class='complete']")),
              len(tree.xpath("//tr[@class='running']")),
              len(tree.xpath("//tr[@class='blocked']"))]

    # check if the process is complated
    if status[0] > 0 and status[1] == 0 and status[2] == 0:
        print('process completed. exiting the program...')
        exit(1)

    # check running status
    if status[1] > 0:
        running_msg = tree.xpath("//tr[@class='running']/td")
        running_status = [running_msg[0].text, running_msg[2].text,
                          running_msg[3].text, running_msg[4].text]

        # check if there's any change in running status
        rerun = True
        for i, v in enumerate(prev_status):
            if v != running_status[i]:
                if v is not None:
                    completed += int(running_status[3]) - int(prev_status[3])
                else:
                    completed = int(running_status[3])
                prev_status = running_status
                rerun = False
                if int(completed / 50) != notify_completed:
                    notify_completed = int(completed / 50)
                    sms('status: {}'.format('/'.join(running_status)))
                break
        if rerun:
            sms('killing process: {}\n re-run process...'.format(process.pid))
            process.terminate()
            process = subprocess.Popen(command)
            time.sleep(120)  # wait for server running up
