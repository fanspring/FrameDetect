# -*- coding: utf-8 -*-  

'''
工具类
'''

from numpy import *
import commands
import os
import platform, subprocess, re
from common import log as logg





'''常量
'''
V_BITRATE = 'bitrate' #视频+音频比特率
V_BITRATE_VIDEO = 'bitrate_video' #视频的比特率
V_DURATION = 'duration' #时长
V_DURATION_SECONDS = 'duration_seconds' #时长秒数
V_RESOLUTION = 'resolution' #分辨率
V_FPS = 'fps' #帧率

V_QPMIN ='qMin'  #最小QP
V_QPMAX ='qMax'  #最小QP
V_QPAVG ='qAvg'  #最小QP




class ut():
    _logger = None

    @classmethod
    def set_logger(cls,logger):
        ut._logger = logger

    @classmethod
    def log(cls,msg, *args):
        ut._logger.info(msg, *args)




'''打印、记录LOG（已过期，建议直接用log.info()）
'''
def log(msg, *args):  
    logg.debug(msg, *args)
def log_info(msg, *args):  
    logg.info(msg, *args)

'''运行shell命令
'''
def run_command(cmd):  
    #os.system方法，获取不到结果
    #os.system(command)
    log('run command now: %s', cmd)
    err = ""
    out = ""
    exitcode = ""
    #commands方法，windows上跑不了
    try:
        if get_os_type(cmd) == 'mac os x':
            (exitcode, out) = commands.getstatusoutput(cmd)
        elif get_os_type(cmd) == 'win':
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            exitcode = proc.returncode
        else:
            (exitcode, out) = commands.getstatusoutput(cmd)
    except Exception as e:
        log('runcommand os failed %s' % e)
        exitcode = -1
        out = 'runcommand os failed'

    if exitcode == 0: #运行成功
        return exitcode, out + err
    else:
        log('command run failed, code: %s, errormsg: %s', exitcode, err)
        return exitcode, out + err
    
'''判断系统
'''
def get_os_type(cmd):  
    sysstr = platform.system() 
    if(sysstr == "Windows"):
        return 'win'
    elif(sysstr == "Linux"):
        return 'linux'
    elif(sysstr == "Darwin"):
        return 'mac os x'
    else:
        return 'other'
    
'''解析ffmpeg的输出
'''
def parse_ffmpeg_output(output):
    params = {V_BITRATE: '', V_BITRATE_VIDEO: '', V_DURATION: '', V_RESOLUTION: '', V_FPS: '',V_DURATION_SECONDS:0}
    for line in output.split('\n'):
        #print line
        if 'bitrate' in line and 'Duration' in line: #码率
            tmp = re.findall(r"bitrate: (\d+) kb/s", line)
            if len(tmp) > 0: params[V_BITRATE] = tmp[0]
            tmp = re.findall(r"Duration: (\d+\:\d+\:\d+\.\d+),", line)
            if len(tmp) > 0: 
                params[V_DURATION] = tmp[0]
                params[V_DURATION_SECONDS] = parse_duration(tmp[0])
        if ('Video:' in line) and ('h26' in line or 'avc' in line or 'yuv420' in line): #视频
            tmp = re.findall(r"(\d{3,4}x\d{3,4})[, ]", line)
            if len(tmp) > 0: params[V_RESOLUTION] = tmp[0]
            tmp = re.findall(r"(\d+) kb/s,", line)
            if len(tmp) > 0: params[V_BITRATE_VIDEO] = tmp[0]
            tmp = re.findall(r"(\d+) fps,", line)
            if len(tmp) > 0: params[V_FPS] = tmp[0]
    
    log('parse ffmpeg output result: %s' % params)
    return params
    



'''解析时长
'''
def parse_duration(duration):
    try:
        durationArr = re.findall(r"(\d+)\:(\d+)\:(\d+)\.(\d+)", duration)
        sec = int(durationArr[0][2])
        min = int(durationArr[0][1])
        hour = int(durationArr[0][0])
        seconds = sec + min*60 + hour*60*60
        if seconds > 5*60*60: return -1
        return seconds
    except Exception as e:
        log(e)
        return -1


'''创建目录
'''
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
'''清空目录，不删除文件夹
'''
def clear_dir(src):
    try:
        if os.path.isfile(src):
            os.remove(src)
        elif os.path.isdir(src):
            for item in os.listdir(src):
                itemsrc = os.path.join(src,item)
                clear_dir(itemsrc) 
    except Exception as e:
        log(e)
        pass

if __name__ == '__main__':

    pass



