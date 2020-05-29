import os

os.system("sed '/^model.add(Dense(units=10,.*/i model.add(Dense(units=128, activation=relu))' /dockerfiles/train.py>/dockerfiles/model.txt")

f=open("/dockerfiles/model.txt",'r')
filedata=f.read()
f.close()
newdata = filedata.replace('relu)','"relu")')

f=open("/dockerfiles/model.txt",'w')
f.write(newdata)
f.close()
os.system('mv /dockerfles/model.txt /dockerfiles/train.py')