# GSMsecuritysystem
Final Year Project of a GSM security system with machine learning


Firstly, real world testing was done followed by exploritory data analysis to pick distinct features of a security breach. 
![image](https://user-images.githubusercontent.com/32158774/170492490-d99e90ff-cf57-461c-ac3b-0840ad4e34d0.png)

Findings: 
![image](https://user-images.githubusercontent.com/32158774/170492627-2ab18d82-7469-4b2b-8a9b-9f5e39835f8f.png)
Normal  movement

![image](https://user-images.githubusercontent.com/32158774/170492681-8865d8fb-028c-43b0-a278-3425145eb298.png)
Security breach


Below is the workflow of the system.
1. An accelerometer reads in user data to the raspberry pi where an ID3 model learns distinctive features from the data.
2. When data is trained, the data is read from the accelerometer where a python script decides whether there is a security breach
3. An sms message is sent to the user to alert them of a security breach.
![image](https://user-images.githubusercontent.com/32158774/170491720-214e4468-a572-43b2-9a0c-409ea894d7d1.png)


Here is the design
![image](https://user-images.githubusercontent.com/32158774/170492288-b3dc2af3-4e59-4ef8-a8a5-179b1cfc6798.png)
