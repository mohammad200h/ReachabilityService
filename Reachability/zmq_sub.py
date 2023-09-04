import zmq

context = zmq.Context()

sub_one = context.socket(zmq.SUB)
sub_one.connect('tcp://127.0.0.1:2000')
sub_one.setsockopt_string(zmq.SUBSCRIBE,'')

sub_two = context.socket(zmq.SUB)
sub_two.connect('tcp://127.0.0.1:2001')
sub_two.setsockopt_string(zmq.SUBSCRIBE,'')


while True:
    msg = sub_one.recv_string()
    print(msg)
    msg = sub_two.recv_string()
    print(msg)