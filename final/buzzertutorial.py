#Libraries
import RPi.GPIO as GPIO
from time import sleep
#Disable warnings (optional)
GPIO.setwarnings(False)
#Select GPIO mode
GPIO.setmode(GPIO.BCM)
#Set buzzer - pin 23 as output
buzzer=23 
GPIO.setup(buzzer,GPIO.OUT)
#Run forever loop
def buzz():
    c=0
    while True:
        GPIO.output(buzzer,GPIO.HIGH)
        sleep(0.5) # Delay in seconds
        GPIO.output(buzzer,GPIO.LOW)
        sleep(0.5)
        c=c+1
        if(c>4):
            break
    
    