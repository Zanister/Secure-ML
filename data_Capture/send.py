import pika
import time
from data_Capture.tcpdump_capture import TcpDump

if __name__ == "__main__":
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='logs', exchange_type='fanout')

    while True:
        filename = f"dump-{int(time.time())}.pcap"
        capture = TcpDump(filename)
        capture.start(duration=60, iface='eth1')
        time.sleep(60)  # Wait for capture to finish
        capture.stop()

        channel.basic_publish(exchange='logs', routing_key='', body=filename)
        print(f"[x] Sent {filename} to RabbitMQ")
