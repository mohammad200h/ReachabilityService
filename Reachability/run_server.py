#!/usr/bin/env python
from com import Server 
from message import Position


def test_task(a):
    return Position(2,2,2)
    


server =Server(test_task)
server.start()





