#encoding:utf-8
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import datetime
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def report(content_map, subject="report", receivers= ['chenxinhua@linkface.cn']):

    config = {
            'address': 'smtp.exmail.qq.com',
            'port': '25',
            'domain': 'linkface.cn',
            'user_name': 'notify@linkface.cn',
            'password': '3UtFtREb2AoAkCkd',
            'encode': 'utf-8'
            }

    sender = config['user_name']
    # receivers = ['AIG@linkface.cn']  # cloudserviceexception@linkface.cn
    datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    content_str = ""

    content_str  += "<table class='gridtable'><tr><th>关键值</th><th>结果</th></tr>"
    for key,value in content_map.items():
        content_str += "<tr><td>"+str(key)+"</td><td>"+str(value)+"</td></tr>"
    content_str +="</table>"
    content = '''
            <!DOCTYPE html>
            <html>
            <head>
              <meta content="text/html; charset=UTF-8" http-equiv="Content-Type="/>
              <style>
                body{font-size:14px;line-height:1.5;background-color:#ececec;font-family:"Avenir-Book", Helvetica, "Helvetica Neue", Arial, Geneva,sans-serif;background-color:#ececec;text-align:left}
                .content-holder{max-width:640px;margin:0px auto;padding:20px 25px;background-color:#fff;box-shadow:0 2px 2px rgba(0,0,0,0.3);dmargin-top:40px;border-radius:6px}
                .subtitle{margin-bottom:30px} .content{font-size:18px;color:#000000}
                table.gridtable {
                    font-family: verdana,arial,sans-serif;
                    font-size:11px;
                    color:#333333;
                    border-width: 1px;
                    border-color: #666666;
                    border-collapse: collapse;
                }
                table.gridtable th {
                    border-width: 1px;
                    padding: 8px;
                    border-style: solid;
                    border-color: #666666;
                    background-color: #dedede;
                }
                table.gridtable td {
                    border-width: 1px;
                    padding: 8px;
                    border-style: solid;
                    border-color: #666666;
                    background-color: #ffffff;
                }
              </style>
            </head>
            <body>
            <div>
                <h3>实验结果统计：</h3>

                

                <div>''' + content_str + '''</div>
                <br>

                <div>如有问题，请及时反馈，发送时间：''' + datetime + '''</div>
                <br>
                </div>
            </div>
            </body>
            </html>
            '''
    message = MIMEText(content, 'html', config['encode'])
    message['From'] = sender
    message['To'] = ''.join(receivers)

    #subject = '实验结果 ('+datetime+')'
    message['Subject'] = Header(subject, 'utf-8')

    # create smtp object and send mail
    try:
        smtp = smtplib.SMTP()
        smtp.connect(config['address'], config['port'])
        smtp.login(config['user_name'], config['password'])
        smtp.sendmail(sender, receivers, message.as_string())
        print "Send mail success"
    except smtplib.SMTPException:
        print "Send Mail Error"



if __name__ == '__main__':
    contentMap = {"a": 1, "b": 2}
    report(contentMap)
