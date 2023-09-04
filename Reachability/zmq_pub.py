import zmq

context = zmq.Context()
#client will publish to server
pub_one =  context.socket(zmq.PUB)
pub_one.bind('tcp://127.0.0.1:2000')


pub_two =  context.socket(zmq.PUB)
pub_two.bind('tcp://127.0.0.1:2001')


while True:
    pub_one.send_string("pub_one::suck on these balls!")
    pub_two.send_string("pub_two::suck on these balls!")