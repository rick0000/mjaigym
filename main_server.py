import argparse
from mjaigym.tcp.server import Server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48000)
    parser.add_argument("--clientlimit", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./tcplog")

    args = parser.parse_args()
    server = Server(args.port, args.clientlimit, args.logdir)
    server.run()
