import zmq
from message import Oreintation
# https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
context = zmq.Context()
#client will publish to server
service =  context.socket(zmq.REP)
service.bind('tcp://127.0.0.1:2000')



while True:
   req =  service.recv_pyobj()
   print("req:: ",req)
   print(req.x,req.y,req.z)
   res = Oreintation(2,2,2)
   service.send_pyobj(res)
    