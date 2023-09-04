import zmq
from message import Position

context = zmq.Context()

client = context.socket(zmq.REQ)
client.connect('tcp://127.0.0.1:2000')



while True:
    pos = Position(1,1,1)
    req = pos
    client.send_pyobj(pos)
    res = client.recv_pyobj()
    print("res:: ",res)
    print(res.r,res.p,res.y)
    
   
