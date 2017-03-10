#!/usr/bin/env python
'''

 Sigma Koki (OptoSigma)'s stage control module.

 Coded by Kenichiro Tanaka, Jul. 2014. 
   Yagi-lab., ISIR, Osaka University
   Mukaigawa-lab., Nara Institute of Science and Technology

 Dependency
 pyserial: $ sudo pip install pyserial

 Before using this program, add the user to the dialout group:
  $ sudo gpasswd -a username dialout
'''

import serial
import time

# Controller settings
#  in this case, the controller is SHOT-302GS / SHOT-304GS
#
baudrate = 9600
parity = 'N'
databit = 8
stopbit = 1
rtscts = True
portname = '/dev/ttyUSB0'
read_timeout = 1
write_timeout = 1
comm_ack_is_main = True
interval = 2

# Setting lists
baudrate_list = (2400, 4800, 9600, 19200)
step_divisions = (1,2,4,5,8,10,20,25,40,50,80,100,125,200,250)
# Resolutions table (micro meters or sub-micro(1/10) degree per pulse)(half) 
resolutions = [
   ('SGSP46-500(X)', 10),
   ('SGSP33-200(Z)', 6),
   ('SGSP33-200(X)', 6),
   ('SGSP26-200(X)', 2),
   ('SGSP-120YAW', 25) ]

# Verbose mode. print status for each operation.
verbose = True
verbose_level = 1
def __state(msg, level=1):
   if level > verbose_level:
      return
   if verbose:
      print msg

def open():
   '''
   Open a serial port and return the serial object.
   
   Returns
   -------
   serial : serial
       Opened serial object.
   '''
   __state(reduce(lambda x,y: x + ' ' + y, map(str,('open',portname,baudrate,databit,parity,stopbit))), level=2)
   return serial.Serial(port  = portname,
                 baudrate     = baudrate,
                 bytesize     = databit,
                 parity       = parity,
                 stopbits     = stopbit,
                 timeout      = read_timeout,
                 writeTimeout = write_timeout,
                 rtscts       = rtscts)


def send(serial, command):
   '''
   Send command. If COMM/ACK is MAIN, receive the response.
   
   Parameters
   ----------
   serial : serial
       Serial object.
   command : string
       Send command.
       
   Returns
   -------
   response : string
       Response from the controller if COMM/ACK is MAIN, otherwise None.
   '''
   serial.write(command + '\r\n')
   if comm_ack_is_main: #TODO: if command is like Q:, receive the data even the mode is SUB.
      response = serial.readline()[:-2]
      __state(command + ' >> ' + response, level=2)
      return response
   __state('[SUB] ' + command, level=2)


def wait_for_ready(serial):
   '''
   Wait for ready. Check if the controller is operating the stage and when the controller finish the operation, exit the waiting-loop. The check call is done by the interval of self.interval..
   
   Parameters
   ----------
   serial : serial
       Serial object.
       
   Returns
   -------
   None
   '''
   __state('Waiting')
   while True:
      serial.write('!:\r\n')
      ack3 = serial.readline()[:-2]
      __state('!: >> ' + ack3, level=2)
      if ack3.startswith('R'):
         return
      __state('     sleep ' + str(interval), level=2)
      time.sleep(interval)

# return '+' or '-'
def __direction(pulse):
   if pulse < 0:
      return '-'
   return '+'


def move(serial, pulse, stage=1):
   '''
   Move a single stage and wait for operation complete.
   
   Parameters
   ----------
   serial : serial 
       Serial object.
   pulse : int
       Moving pulses.
   stage : int, optional
       Target stage number.
   
   Returns
   -------
   None
   
   See also
   --------
   moves, resume
   '''
   d = __direction(pulse)
   __state('Sending move command')
   send(serial, 'M:' + str(stage) + d + 'P' + str(abs(pulse)))
   send(serial, 'G:')
   wait_for_ready(serial)


def moves(serial, pulse_set):
   '''
   Move multiple stages and wait for operation complete.
   
   Parameters
   ----------
   serial : serial
       Serial object.
   pulse_set : array of int
       Moving pulses for each stage.
       
   Returns
   -------
   None
   
   See also
   --------
   move, resume
   '''
   __state('Sending move command')
   com = ['M:W']
   for pulse in pulse_set:
      com.append(__direction(pulse))
      com.append('P')
      com.append(str(abs(pulse)))
   command = ''.join(com)
   send(serial, command)
   send(serial, 'G:')
   wait_for_ready(serial)


def jog(serial, stage=1, direction=1):
    '''Jog drive
    '''
    com = 'J:' + str(stage) + __direction(direction)
    send(serial, com)
    send(serial, 'G:')
    
def stop(serial, stage=1):
    '''Stop a stage slowly.
    '''
    send(serial, 'L:'+str(stage))

def get_status(serial):
    ''' Get controller's status.
    
    Parameters
    ----------
    serial : serial
        Serial object.
        
    Returns
    -------
    positions : array(2) or array(4), dtype=int
        Position of all stages.
    ack1 : str
        One of (K, X)
        'X' represents command or parameter error. 'K' represents command is accepted successfully.
    ack2 : str
        One of (K, W, R, L, M, 1, 2, 3, 4, E).
        [Common] 'W': all stages are stopped at limit sensor. 'K': all stages stopped normally. 'R': alert stop. Details can be obtained by 'I' command.
        [SHOT-302GS] 'L': 1st stage is stopped at limit sensor. 'M': 2nd stage is stopped at limit sensor.
        [SHOT-304GS] '1' to '4': corresponding stage is stopped at limit sensor. 'E': stages #2 to #4 are stopped at limit sensor.
    ack3 : str
        One of (B, R).
        'B' represents busy, and 'R' represents ready.
    '''
    serial.write('Q:\r\n')
    data = serial.readline()[:-2]
    __state('Q >> ' + data, level=2)
    data = data.split(',')
    ack1, ack2, ack3 = data[-3:]
    positions = map(int, data[:len(data) - 3])
    return positions, ack1, ack2, ack3
    
def is_stopped_at_limit_sensor_or_ready(serial, stage):
    '''Check if the stage is stopped at limit sensor or not.
    '''
    pos, ack1, ack2, ack3 = get_status(serial)
    if ack3 == 'R':
        return True
    if ack2 == 'W':
        return True
    if ack2 == 'K' or 'R':
        return False
    if ack2 == str(stage):
        return True
    if stage==1:
        return ack2 == 'L'
    if stage==2:
        return ack2 == 'M' or ack3 == 'E'
    return ack2 == 'E'

def resume(serial, stages=1):
   '''
   Move all stages to the electrical zero point.
   
   Parameters
   ----------
   serial : serial
       Serial object.
   stages : int
       Number of stages. The stage more than this value does not be resumed its position.
       
   Results
   -------
   None
   '''
   __state('Resuming to electorical zero point')
   com = ['A:W']
   for n in range(stages):
      com.append('+P0')
   command = ''.join(com)
   send(serial, command)
   send(serial, 'G:')
   wait_for_ready(serial)
   
def mechanical_resume(serial):
    '''
    Move all stages to the mechanical zero point, and reset the logical zero point.
    
    Parameters
    ----------
    serial : serial
        Serial object.
    '''
    __state('Mechanical resume')
    command ='H:W'
    send(serial, command)
    wait_for_ready(serial)
    
def mechanical_resume_single(serial, stage_num=1):
    '''
    Move all stages to the mechanical zero point, and reset the logical zero point.
    
    Parameters
    ----------
    serial : serial
        Serial object.
    '''
    __state('Mechanical resume')
    command ='H:' + str(stage_num)
    send(serial, command)
    wait_for_ready(serial)
    
def set_drive_speed(serial, stage=1, low=100, high=10000, acc_time=100):
    '''
    Set the drive speed of the stage.
    
    Parameters
    ----------
    serial : serial
        Serial object.
    stage : int, optional
        Target stage.
    low : int, optional
        Minimum speed of the stage (pulse per second).
    high : int, optional
        Maximum speed of the stage. (pulse per sec.)
    acc_time : int, optional
        Acceralation and deceleration time. (ms)
    
    '''
    command = 'D:' + str(stage) + 'S' + str(low) + 'F' + str(high) + 'R' + str(acc_time)
    send(serial, command)
    #wait_for_ready(serial)

def emergency_stop(serial):
    '''
    Stop all stages immediately. All operations will be aborted.
    '''
    command = 'L:E'
    send(serial, command)
    
def change_step_division(stage=1, dev=2):
    '''
    Change the resolution of the pulse.
    Resolutions (micro-meters per pulse) written in the catalog correspond to this value to be 1 (Full) and 2 (Half).
    
    Parameters
    ----------
    stage : int
        Target stage
    dev : int
        Sub-steps per pulse. This value must be one of step_divisions, which is (1,2,4,5,8,10,20,25,40,50,80,100,125,200,250).
    '''
    if not dev in step_divisions:
        print 'Warning: ' +str(dev) + 'is not a valid division value. -- Ignored.'
        return
    command = 'S:'+str(stage)+str(dev)
    send(serial, command)

# Test this module.
if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser(description='Move stages sample.')
   parser.add_argument('--level', default=10, type=int, help='Verbose level')
   parser.add_argument('-e', '--emergency', action='store_true', help='Emergency stop.')
   parser.add_argument('-r', '--resume', action='store_true', help='Resume all stages to mechanical zero point.')
   parser.add_argument('-m', '--move', type=int, nargs=2, default=(0,0), help='Move stage N for M steps.')
   parser.add_argument('--multimove', type=int, nargs='*', help='Move stages.')
   args = parser.parse_args()

   verbose_level = args.level
   serial_port = open()
   
   if args.emergency:
       emergency_stop(serial_port)
#       return
   
   if args.resume:
       mechanical_resume(serial_port)
#       return
   
   if args.move[0] > 0:
       move(serial_port, args.move[1], args.move[0])
       
   if args.multimove is not None:
       if len(args.multimove) > 1:
           set_drive_speed(serial_port, 1, 1000,10000,1000)
           set_drive_speed(serial_port, 2, 1000,20000,1000)
           set_drive_speed(serial_port, 3, 1000,20000,1000)
           moves(serial_port, args.multimove)
           
#       return
     
#   move(serial_port, 4166, 1) # = 25mm
   
#   set_drive_speed(serial_port, 3)
#   move(serial_port, 14000, 3) # = 84mm
#   move(serial_port, -14000, 3) # = 84mm
   
#   move(serial_port, 10000, 2)
#   resume(serial_port, 3)
   
   # Set here to the origin
   # send(serial, 'R:1')
   # Move stage1 to 10000 pulse # in case of SGSP46-500, it is 10cm.
   #send(serial, 'M:1+P10000') 
   #send(serial, 'G:')
   # Wait the operation
   #wait_for_ready(serial)
   # Move stage1 to -10000 pulse
   #send(serial, 'M:1-P10000')
   #send(serial, 'G:')
   
   
   
