# -*- coding: utf-8 -*-


import httplib, urllib
import json
import time
from FDConfig import logger,FDConfig

IP = "100.97.142.172"
REQUEST_STR = "/service/select_record_frame_multiple.php"
REPORT_STR = "/service/add_record_frame_multiple.php"

REPORT_TABLE = 'frame_detect_result'


REQUEST_COUNT = 20





def do_report(appid,index,values):
    params = urllib.urlencode({'appid': appid,
                               'async': 0,
                               'tablename': REPORT_TABLE,
                               'index': index,
                               'values': values})
    httpClient = None
    try:
        headers = {"Content-type": "application/x-www-form-urlencoded"
            , "Accept": "text/plain"}
        httpClient = httplib.HTTPConnection(IP, 80, timeout=5)
        httpClient.request("POST", REPORT_STR, params, headers)
        response = httpClient.getresponse()
        if(response.status == 200):
            logger.info('[do report],connect success: %s'%response.read())
        else:
            logger.info('[do report],connect failed: %s' % response.status)

    except Exception, e:
        logger.info(e)
    finally:
        if httpClient:
            httpClient.close()

'''report interface
'''
def report_result(report_infos):
    if not len(report_infos) > 0:
        logger.info('[do report] no result data for reporting')
        return

    appid = report_infos[0]['appid'] if report_infos[0].has_key('appid') else 0

    valuelist = []
    for info in report_infos:
        # delete key 'id'
        if info.has_key('id'): del info['id']

        # add ' to report str, otherwise get error
        for key in info.keys():
            if isinstance(info[key],str) or isinstance(info[key],unicode):
                info[key] = "'%s'"%info[key]

        tmp_str = '(%s)'%','.join(info.values())
        valuelist.append(tmp_str)
    values = ','.join(valuelist)

    index = "(%s)"%','.join(report_infos[0].keys())

    do_report(appid=appid, index=index, values=values)
    logger.info('[do report] report data num: %d'%len(report_infos))

def do_geturls(appid, tablename, count):
    params = urllib.urlencode({'appid': appid,
                               'tablename': tablename,
                               'count': count})
    data = None
    httpClient = None
    try:
        headers = {"Content-type": "application/x-www-form-urlencoded"
            , "Accept": "text/plain"}
        httpClient = httplib.HTTPConnection(IP, 80, timeout=5)
        httpClient.request("POST", REQUEST_STR, params, headers)
        response = httpClient.getresponse()
        if(response.status == 200):
            logger.info('[get url infos] appid[%s] connect success: %s'%(appid, response.status))
            data = response.read()
        else:
            logger.info('[get url infos] appid[%s] connect failed: %s' % (appid, response.status))

    except Exception, e:
        logger.info(e)
    finally:
        if httpClient:
            httpClient.close()

    return data

'''get urls interface
'''
def get_pic_infos():
    infos = []
    table = time.strftime("table_%Y%m%d", time.localtime())
    #table = 'table_test'
    response_data = do_geturls(appid=FDConfig._appid,tablename=table,count=REQUEST_COUNT)
    if response_data:
        response_dict = json.loads(response_data)
        if response_dict.has_key('data'):
            data_list = json.loads(response_dict['data'])
            infos = data_list
    if type(infos) == list:
        logger.info('[get url infos] get url infos num: %d' % len(infos))
        return infos
    else:
        logger.warning('response type error')
        logger.warning(response_data)
        return []





if __name__ == '__main__':
    do_report('','','')



