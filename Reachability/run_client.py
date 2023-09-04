#!/usr/bin/env python
from com import Client
from message import Position
from time import sleep





client = Client()

while 1:
    
    p = Position(1,1,1)
    b = client.send(p)
    print(p)
    sleep(0.5)
# print(b.x,p.y,p.z)

